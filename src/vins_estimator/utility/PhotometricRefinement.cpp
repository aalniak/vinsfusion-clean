#include <vins_estimator/utility/PhotometricRefinement.h>
#include <iostream>
#include <vins_estimator/factor/pose_local_parameterization.h>
#include <vins_estimator/factor/PhotometricSparseFactor.h>
#include <ceres/ceres.h>
#include <glog/logging.h>

namespace vins::estimator {

PhotometricRefinement::PhotometricRefinement() : keep_running_(true) {
    processing_thread_0_ = std::thread(&PhotometricRefinement::threadLoop0, this);
    processing_thread_1_ = std::thread(&PhotometricRefinement::threadLoop1, this);
}

PhotometricRefinement::~PhotometricRefinement() {
    keep_running_ = false;
    task_cond_0_.notify_all();
    task_cond_1_.notify_all();
    if (processing_thread_0_.joinable()) {
        processing_thread_0_.join();
    }
    if (processing_thread_1_.joinable()) {
        processing_thread_1_.join();
    }
}

void PhotometricRefinement::submitTask(double t_ref, double t_cur,
                                       const cv::Mat& img_ref, const cv::Mat& img_cur,
                                       const cv::Mat& depth_ref, const cv::Mat& depth_cur,
                                       const Eigen::Matrix3d& K,
                                       const Eigen::Quaterniond& q_initial, const Eigen::Vector3d& t_initial,
                                       double init_scale, double init_shift,
                                       double init_scale_cur, double init_shift_cur) {
    
    // Determine which queue to push to
    // Simple load balancing: check queue sizes
    size_t q0_size = 0;
    size_t q1_size = 0;
    
    {
        std::lock_guard<std::mutex> lock(task_mutex_0_);
        q0_size = task_queue_0_.size();
    }
    {
        std::lock_guard<std::mutex> lock(task_mutex_1_);
        q1_size = task_queue_1_.size();
    }
    
    // Prepare task
    Task task;
    task.t_ref = t_ref;
    task.t_cur = t_cur;
    task.img_ref = img_ref.clone();
    task.img_cur = img_cur.clone();
    task.depth_ref = depth_ref.clone();
    task.depth_cur = depth_cur.empty() ? cv::Mat() : depth_cur.clone();
    task.K = K;
    task.q_initial = q_initial;
    task.t_initial = t_initial;
    task.init_scale = init_scale;
    task.init_shift = init_shift;
    task.init_scale_cur = init_scale_cur;
    task.init_shift_cur = init_shift_cur;

    // If a thread is not busy, prioritize it (though queue size should be 0)
    // Otherwise pick the shorter queue
    bool use_t0 = (q0_size <= q1_size);
    /*
    if (!t0_busy) use_t0 = true;
    else if (!t1_busy) use_t0 = false;
    */
   
    if (use_t0) {
        std::lock_guard<std::mutex> lock(task_mutex_0_);
        task_queue_0_.push(task);
        task_cond_0_.notify_one();
        //printf("[Refinement] Submitted to Thread 0 (q=%zu)\n", q0_size);
    } else {
        std::lock_guard<std::mutex> lock(task_mutex_1_);
        task_queue_1_.push(task);
        task_cond_1_.notify_one();
        //printf("[Refinement] Submitted to Thread 1 (q=%zu)\n", q1_size);
    }
}

void PhotometricRefinement::clearQueues() {
    int discarded = 0;
    {
        std::lock_guard<std::mutex> lock(task_mutex_0_);
        while (!task_queue_0_.empty()) {
            task_queue_0_.pop();
            discarded++;
        }
    }
    {
        std::lock_guard<std::mutex> lock(task_mutex_1_);
        while (!task_queue_1_.empty()) {
            task_queue_1_.pop();
            discarded++;
        }
    }
    if (discarded > 0) {
        printf("[Refinement] Cleared %d pending tasks from queues.\n", discarded);
    }
}

bool PhotometricRefinement::getResult(RefinementResult& result) {
    // Check thread 0 results first
    {
        std::lock_guard<std::mutex> lock(result_mutex_0_);
        if (!result_queue_0_.empty()) {
            result = result_queue_0_.front();
            result_queue_0_.pop();
            return true;
        }
    }
    
    // Check thread 1 results
    {
        std::lock_guard<std::mutex> lock(result_mutex_1_);
        if (!result_queue_1_.empty()) {
            result = result_queue_1_.front();
            result_queue_1_.pop();
            return true;
        }
    }
    
    return false;
}

// ============================================================
// HAND-ROLLED GAUSS-NEWTON PHOTOMETRIC SOLVER
// Analytic Jacobians, Huber IRLS, SE(3) left perturbation.
// Runs 10-20x faster than Ceres-based approach.
// ============================================================
/* PhotometricRefinement::GNResult PhotometricRefinement::solvePhotometricGN(
    const std::vector<FeaturePoint>& features,
    const cv::Mat& img_cur,
    const Eigen::Matrix3d& K,
    const Eigen::Quaterniond& q_init,
    const Eigen::Vector3d& t_init,
    int max_iterations)
{
    GNResult result;
    result.H_pose.setZero();
    
    const double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    const int W = img_cur.cols, H = img_cur.rows;
    
    printf("[GN DEBUG] N=%zu, img=%dx%d type=%d, fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n",
           features.size(), W, H, img_cur.type(), fx, fy, cx, cy);
    if (!features.empty()) {
        printf("[GN DEBUG] feat[0]: u=(%.1f,%.1f) intensity=%.1f inv_depth=%.4f\n",
               features[0].u.x(), features[0].u.y(), features[0].intensity, features[0].inv_depth);
    }
    printf("[GN DEBUG] q_init: w=%.4f xyz=(%.4f,%.4f,%.4f) t_init=(%.4f,%.4f,%.4f)\n",
           q_init.w(), q_init.x(), q_init.y(), q_init.z(),
           t_init.x(), t_init.y(), t_init.z());
    
    // Convert to grayscale if needed
    cv::Mat img_gray;
    if (img_cur.channels() == 3) {
        cv::cvtColor(img_cur, img_gray, cv::COLOR_BGR2GRAY);
    } else {
        img_gray = img_cur;
    }
    
    // Precompute image gradients (central differences via Sobel ksize=1)
    cv::Mat grad_x, grad_y;
    cv::Sobel(img_gray, grad_x, CV_16S, 1, 0, 1);
    cv::Sobel(img_gray, grad_y, CV_16S, 0, 1, 1);
    
    // Precompute 3D reference points (backproject once)
    const size_t N = features.size();
    std::vector<Eigen::Vector3d> pts_ref(N);
    for (size_t i = 0; i < N; i++) {
        double d = 1.0 / features[i].inv_depth;
        double xn = (features[i].u.x() - cx) / fx;
        double yn = (features[i].u.y() - cy) / fy;
        pts_ref[i] = Eigen::Vector3d(xn * d, yn * d, d);
    }
    
    // Current pose estimate (left perturbation convention)
    Eigen::Matrix3d R_cur = q_init.toRotationMatrix();
    Eigen::Vector3d t_cur = t_init;
    
    double last_cost = std::numeric_limits<double>::max();
    int actual_iters = 0;
    
    // Save best pose for revert
    Eigen::Matrix3d R_best = R_cur;
    Eigen::Vector3d t_best = t_cur;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        Eigen::Matrix<double, 6, 6> H_acc = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b_acc = Eigen::Matrix<double, 6, 1>::Zero();
        double total_cost = 0;
        int num_valid = 0;
        
        for (size_t i = 0; i < N; i++) {
            // Transform reference point to current frame
            Eigen::Vector3d Pc = R_cur * pts_ref[i] + t_cur;
            
            double Z = Pc.z();
            if (Z <= 0.01) continue;
            
            double X = Pc.x(), Y = Pc.y();
            double Zinv = 1.0 / Z;
            double Zinv2 = Zinv * Zinv;
            
            // Project to pixel
            double u_proj = fx * X * Zinv + cx;
            double v_proj = fy * Y * Zinv + cy;
            
            // Bounds check (1px margin for gradient + interpolation)
            if (u_proj < 1.0 || u_proj >= W - 1.0 || v_proj < 1.0 || v_proj >= H - 1.0)
                continue;
            
            // Bilinear interpolation indices
            int u0 = (int)u_proj, v0 = (int)v_proj;
            double du = u_proj - u0, dv = v_proj - v0;
            double w00 = (1-du)*(1-dv), w10 = du*(1-dv), w01 = (1-du)*dv, w11 = du*dv;
            
            // Interpolate intensity
            const uchar* row0 = img_gray.ptr<uchar>(v0);
            const uchar* row1 = img_gray.ptr<uchar>(v0 + 1);
            double I_cur_val = w00*row0[u0] + w10*row0[u0+1] + w01*row1[u0] + w11*row1[u0+1];
            
            // Interpolate gradients
            const short* gx0 = grad_x.ptr<short>(v0);
            const short* gx1 = grad_x.ptr<short>(v0 + 1);
            const short* gy0 = grad_y.ptr<short>(v0);
            const short* gy1 = grad_y.ptr<short>(v0 + 1);
            double Ix = w00*gx0[u0] + w10*gx0[u0+1] + w01*gx1[u0] + w11*gx1[u0+1];
            double Iy = w00*gy0[u0] + w10*gy0[u0+1] + w01*gy1[u0] + w11*gy1[u0+1];
            
            // Residual
            double r = I_cur_val - features[i].intensity;
            
            // Huber IRLS weight
            double abs_r = std::abs(r);
            double w = 1.0;
            if (abs_r > huber_loss_) {
                w = huber_loss_ / abs_r;
            }
            
            // Jacobian: J = [Ix, Iy] · J_proj (1×6)
            // J_proj = [fx/Z    0      -fx·X/Z²   -fx·XY/Z²     fx·(1+X²/Z²)   -fx·Y/Z ]
            //          [0       fy/Z   -fy·Y/Z²   -fy·(1+Y²/Z²) fy·XY/Z²       fy·X/Z  ]
            Eigen::Matrix<double, 1, 6> J;
            J(0) = Ix * fx * Zinv;                                          // dt_x
            J(1) = Iy * fy * Zinv;                                          // dt_y
            J(2) = -(Ix * fx * X + Iy * fy * Y) * Zinv2;                   // dt_z
            J(3) = -(Ix * fx * X * Y * Zinv2 + Iy * fy * (1.0 + Y*Y*Zinv2)); // dw_x
            J(4) = Ix * fx * (1.0 + X*X*Zinv2) + Iy * fy * X * Y * Zinv2;   // dw_y
            J(5) = (-Ix * fx * Y + Iy * fy * X) * Zinv;                     // dw_z
            
            // Accumulate normal equations
            H_acc.noalias() += w * J.transpose() * J;
            b_acc.noalias() += w * J.transpose() * r;
            
            // Huber cost
            if (abs_r <= huber_loss_) {
                total_cost += 0.5 * r * r;
            } else {
                total_cost += huber_loss_ * (abs_r - 0.5 * huber_loss_);
            }
            num_valid++;
        }
        
        if (iter == 0) {
            result.initial_cost = total_cost;
            printf("[GN DEBUG] iter 0: num_valid=%d/%zu, total_cost=%.4e\n", num_valid, N, total_cost);
            if (num_valid > 0) {
                printf("[GN DEBUG] H diag: %.2e %.2e %.2e %.2e %.2e %.2e\n",
                       H_acc(0,0), H_acc(1,1), H_acc(2,2), H_acc(3,3), H_acc(4,4), H_acc(5,5));
                printf("[GN DEBUG] b: %.2e %.2e %.2e %.2e %.2e %.2e\n",
                       b_acc(0), b_acc(1), b_acc(2), b_acc(3), b_acc(4), b_acc(5));
            }
        }
        actual_iters = iter + 1;
        
        // Cost increase → revert to best and stop
        if (iter > 0 && total_cost >= last_cost) {
            R_cur = R_best;
            t_cur = t_best;
            break;
        }
        
        last_cost = total_cost;
        R_best = R_cur;
        t_best = t_cur;
        result.H_pose = H_acc;
        result.num_valid = num_valid;
        
        // Solve H · δξ = -b
        Eigen::Matrix<double, 6, 1> dx = H_acc.ldlt().solve(-b_acc);
        
        if (dx.hasNaN() || dx.norm() > 0.5) break; // Sanity: max 0.5m/0.5rad per step
        
        // SE(3) left perturbation update: T' = exp(δξ) · T
        Eigen::Vector3d dt_step = dx.head<3>();
        Eigen::Vector3d dw = dx.tail<3>();
        
        double theta = dw.norm();
        Eigen::Matrix3d dR;
        if (theta < 1e-10) {
            dR = Eigen::Matrix3d::Identity();
        } else {
            Eigen::AngleAxisd aa(theta, dw / theta);
            dR = aa.toRotationMatrix();
        }
        
        t_cur = dR * t_cur + dt_step;
        R_cur = dR * R_cur;
        
        // Convergence check
        if (dt_step.norm() < 1e-7 && theta < 1e-7) break;
    }
    
    result.q_cur_ref = Eigen::Quaterniond(R_cur).normalized();
    result.t_cur_ref = t_cur;
    result.final_cost = last_cost;
    result.iterations = actual_iters;
    
    return result;
} */

PhotometricRefinement::GNResult PhotometricRefinement::solvePhotometricGN_Extended(
    const std::vector<FeaturePoint>& features,
    const cv::Mat& img_cur,
    const Eigen::Matrix3d& K,
    const Eigen::Quaterniond& q_init,
    const Eigen::Vector3d& t_init,
    int max_iterations,
    bool opt_affine,
    bool opt_scaleshift,
    const double* init_affine,
    const double* init_scaleshift,
    bool opt_shift)
{
    GNResult result;
    
    // 1. Initialize State
    Eigen::Vector3d t_cur = t_init;
    Eigen::Matrix3d R_cur = q_init.toRotationMatrix();
    double alpha = (init_affine) ? init_affine[0] : 1.0;
    double beta  = (init_affine) ? init_affine[1] : 0.0;
    double scale = (init_scaleshift) ? init_scaleshift[0] : 1.0;
    double shift = (init_scaleshift) ? init_scaleshift[1] : 0.0;

    // Determine Parameter Block Layout
    // 0-5: Pose
    // 6-7: Affine (if enabled)
    // 6 or 8: Scale (depends on affine), optionally +1 for shift
    int idx_pose = 0;
    int idx_aff  = opt_affine ? 6 : -1;
    int idx_ss   = opt_scaleshift ? (opt_affine ? 8 : 6) : -1;

    int dim = 6;
    if (opt_affine) dim += 2;
    if (opt_scaleshift) dim += (opt_shift ? 2 : 1); // scale only, or scale+shift
    
    Eigen::Matrix<double, 1, 10> J; // Max size
    J.setZero();

    const double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    const int W = img_cur.cols, H = img_cur.rows;
    const size_t N = features.size();
    
    printf("[GN-EXT DEBUG] N=%zu, img=%dx%d, fx=%.1f, aff=%d, ss=%d, opt_shift=%d\n",
           N, W, H, fx, opt_affine, opt_scaleshift, opt_shift);
    if (opt_scaleshift) {
        printf("[GN-EXT DEBUG] Init Scale=%.4f Shift=%.4f (%s)\n", scale, shift,
               opt_shift ? "Shift Optimized" : "Shift Fixed");
    }

    // Image Gradients
    cv::Mat img_gray, grad_x, grad_y;
    if (img_cur.channels() == 3) cv::cvtColor(img_cur, img_gray, cv::COLOR_BGR2GRAY);
    else img_gray = img_cur;
    
    cv::Sobel(img_gray, grad_x, CV_16S, 1, 0, 1);
    cv::Sobel(img_gray, grad_y, CV_16S, 0, 1, 1);

    double last_cost = std::numeric_limits<double>::max();
    
    // Backup for rollback
    Eigen::Vector3d t_best = t_cur;
    Eigen::Matrix3d R_best = R_cur;
    
    double alpha_best = alpha, beta_best = beta;
    double scale_best = scale, shift_best = shift;
    int final_num_valid = 0;
    
    // Store best Hessian found so far
    Eigen::Matrix<double, 10, 10> H_best = Eigen::Matrix<double, 10, 10>::Zero();
    
    for (int iter = 0; iter < max_iterations; iter++) {
        Eigen::Matrix<double, 10, 10> H_mat = Eigen::Matrix<double, 10, 10>::Zero();
        Eigen::Matrix<double, 10, 1>  b = Eigen::Matrix<double, 10, 1>::Zero();
        int num_valid = 0;
        double total_cost = 0;
        

        for (size_t i = 0; i < N; i++) {
            // === A. Center projection (once per feature) ===
            double d_raw = features[i].inv_depth;
            double d_new = (opt_scaleshift) ? (scale * d_raw + shift) : d_raw;
            if (d_new <= 1e-4) continue;
            double Z_ref = 1.0 / d_new;

            double xn = (features[i].u.x() - cx) / fx;
            double yn = (features[i].u.y() - cy) / fy;
            Eigen::Vector3d P_ref(xn * Z_ref, yn * Z_ref, Z_ref);
            Eigen::Vector3d P_cur = R_cur * P_ref + t_cur;

            double Z_cur = P_cur.z();
            if (Z_cur <= 0.01) continue;

            double u_proj = fx * P_cur.x() / Z_cur + cx;
            double v_proj = fy * P_cur.y() / Z_cur + cy;

            // === B. Shared geometric factors (once per feature) ===
            double X = P_cur.x(), Y = P_cur.y();
            double Zinv = 1.0 / Z_cur;
            double Zinv2 = Zinv * Zinv;

            // Scale/shift geometric vectors (shared across pattern)
            Eigen::Vector3d dPc_ds, dPc_dh;
            if (opt_scaleshift) {
                Eigen::Vector3d RP_ref = P_cur - t_cur;
                dPc_ds = RP_ref * (-Z_ref * d_raw);
                if (opt_shift) dPc_dh = RP_ref * (-Z_ref);
            }

            // === C. Affine warp matrix: maps ref pixel offset → cur pixel offset ===
            // For offset (du,dv) with constant depth, the 3D offset in cur frame is
            // dP = R * Z_ref * [du/fx, dv/fy, 0], and we project via first-order expansion.
            double A00 = Z_ref * (R_cur(0,0) * Z_cur - X * R_cur(2,0)) * Zinv2;
            double A01 = Z_ref * (fx/fy) * (R_cur(0,1) * Z_cur - X * R_cur(2,1)) * Zinv2;
            double A10 = Z_ref * (fy/fx) * (R_cur(1,0) * Z_cur - Y * R_cur(2,0)) * Zinv2;
            double A11 = Z_ref * (R_cur(1,1) * Z_cur - Y * R_cur(2,1)) * Zinv2;

            // === D. Process center + pattern members ===
            // Total samples: 1 (center) + pattern_count
            int n_samples = 1 + features[i].pattern_count;
            for (int si = 0; si < n_samples; si++) {
                double u_target, v_target, I_ref_val;

                if (si == 0) {
                    // Center sample
                    u_target = u_proj;
                    v_target = v_proj;
                    I_ref_val = features[i].intensity;
                } else {
                    // Pattern member: warp offset through affine matrix
                    const auto& pm = features[i].pattern[si - 1];
                    double du_warp = A00 * pm.du + A01 * pm.dv;
                    double dv_warp = A10 * pm.du + A11 * pm.dv;
                    u_target = u_proj + du_warp;
                    v_target = v_proj + dv_warp;
                    I_ref_val = pm.intensity;
                }

                // Bounds check
                if (u_target < 1.0 || u_target >= W - 1.0 ||
                    v_target < 1.0 || v_target >= H - 1.0) continue;

                // Bilinear interpolation (intensity + gradients)
                int u0 = (int)u_target, v0 = (int)v_target;
                double du_f = u_target - u0, dv_f = v_target - v0;
                double w00 = (1-du_f)*(1-dv_f), w10 = du_f*(1-dv_f);
                double w01 = (1-du_f)*dv_f,     w11 = du_f*dv_f;

                const uchar* row0 = img_gray.ptr<uchar>(v0);
                const uchar* row1 = img_gray.ptr<uchar>(v0+1);
                double I_cur_val = w00*row0[u0] + w10*row0[u0+1] + w01*row1[u0] + w11*row1[u0+1];

                const short* gx0 = grad_x.ptr<short>(v0);
                const short* gx1 = grad_x.ptr<short>(v0+1);
                const short* gy0 = grad_y.ptr<short>(v0);
                const short* gy1 = grad_y.ptr<short>(v0+1);
                double Ix = w00*gx0[u0] + w10*gx0[u0+1] + w01*gx1[u0] + w11*gx1[u0+1];
                double Iy = w00*gy0[u0] + w10*gy0[u0+1] + w01*gy1[u0] + w11*gy1[u0+1];

                // Residual
                double pred_val = (opt_affine) ? (alpha * I_ref_val + beta) : I_ref_val;
                double r = I_cur_val - pred_val;
                double abs_r = std::abs(r);
                double wt = (abs_r > huber_loss_) ? (huber_loss_ / abs_r) : 1.0;

                // Jacobian: per-sample Ix/Iy, shared geometric factors
                Eigen::Matrix<double, 1, 10> J;
                J.setZero();

                double j0 = Ix * fx * Zinv;
                double j1 = Iy * fy * Zinv;
                double j2 = -(Ix * fx * X + Iy * fy * Y) * Zinv2;

                J(0) = j0;
                J(1) = j1;
                J(2) = j2;
                J(3) = -(Ix * fx * X * Y * Zinv2 + Iy * fy * (1.0 + Y*Y*Zinv2));
                J(4) = Ix * fx * (1.0 + X*X*Zinv2) + Iy * fy * X * Y * Zinv2;
                J(5) = (-Ix * fx * Y + Iy * fy * X) * Zinv;

                if (opt_affine) {
                    J(idx_aff)     = -I_ref_val;
                    J(idx_aff + 1) = -1.0;
                }

                if (opt_scaleshift) {
                    J(idx_ss) = j0 * dPc_ds.x() + j1 * dPc_ds.y() + j2 * dPc_ds.z();
                    if (opt_shift) {
                        J(idx_ss + 1) = j0 * dPc_dh.x() + j1 * dPc_dh.y() + j2 * dPc_dh.z();
                    }
                }

                // Accumulate
                auto J_active = J.head(dim);
                H_mat.topLeftCorner(dim, dim).noalias() += wt * J_active.transpose() * J_active;
                b.head(dim).noalias() += wt * J_active.transpose() * r;

                if (abs_r <= huber_loss_) total_cost += 0.5 * r * r;
                else total_cost += huber_loss_ * (abs_r - 0.5 * huber_loss_);

                num_valid++;
            }
        }

        if (iter == 0) {
            result.initial_cost = total_cost;
            printf("[GN-EXT DEBUG] iter 0: residuals=%d (features=%zu), total_cost=%.4e\n", num_valid, N, total_cost);
        }
        result.iterations = iter + 1;
        final_num_valid = num_valid;

        // --- DIAGNOSTICS FOR SCALE ---
        if (opt_scaleshift) {
            // Check average depth and jacobian values
            double avg_d = 0;
            double avg_J_ss = 0;
            double avg_dPc_ds = 0;
            int count = 0;
            
            // Re-run loop for diagnostics (inefficient but safe for debug)
            for (size_t i = 0; i < N; i += 10) { // Sample every 10th
                double d_raw = features[i].inv_depth;
                double d_new = (scale * d_raw); // shift is 0
                if (d_new <= 1e-4) continue;
                double Z_ref = 1.0 / d_new;
                double xn = (features[i].u.x() - cx) / fx;
                double yn = (features[i].u.y() - cy) / fy;
                Eigen::Vector3d P_ref(xn * Z_ref, yn * Z_ref, Z_ref);
                Eigen::Vector3d P_cur = R_cur * P_ref + t_cur;
                Eigen::Vector3d RP_ref = P_cur - t_cur; 
                Eigen::Vector3d dPc_ds = RP_ref * (-Z_ref * d_raw);
                
                // Approximate J for scale (assuming J_pose approx [1,0,0] etc)
                // Just check magnitude of geometric derivative
                avg_d += d_raw;
                avg_dPc_ds += dPc_ds.norm();
                count++;
            }
            if (count > 0) {
                 avg_d /= count;
                 avg_dPc_ds /= count;
                 // Debug prints removed for cleanliness
            }
        }
        // -----------------------------

        // --- ADD REGULARIZATION (Priors) ---
        // Prevents parameter drift when geometry is weak.
        // Adjusted based on user feedback (was "stiff").
        
        // 1. Affine: alpha->1, beta->0
        if (opt_affine) {
            double w_aff = 0.1 * num_valid; // Reduced from 10.0
            
            // Diagnostics for stiffness
            if (iter == 0) {
                 double H_data = H_mat(idx_aff, idx_aff);
                 printf("[GN-EXT DEBUG] Affine Stiff: H_data=%.2e, w_prior=%.2e, Ratio=%.4f\n", 
                        H_data, w_aff, (H_data > 0) ? w_aff/H_data : 0.0);
            }

            // alpha prior
            H_mat(idx_aff, idx_aff) += w_aff;
            b(idx_aff) += w_aff * (alpha - 1.0);
            total_cost += 0.5 * w_aff * (alpha - 1.0) * (alpha - 1.0);
            
            // beta prior
            H_mat(idx_aff + 1, idx_aff + 1) += w_aff;
            b(idx_aff + 1) += w_aff * (beta - 0.0);
            total_cost += 0.5 * w_aff * (beta - 0.0) * (beta - 0.0);
        }

        // 2. Scale prior (pull toward init value)
        if (opt_scaleshift) {
            double w_ss = 1.0;
            double prior_s = init_scaleshift ? init_scaleshift[0] : 1.0;

            if (iter == 0) {
                 double H_data = H_mat(idx_ss, idx_ss);
                 printf("[GN-EXT DEBUG] Scale Stiff: H_data=%.2e, w_prior=%.2e, Ratio=%.4f\n",
                        H_data, w_ss, (H_data > 0) ? w_ss/H_data : 0.0);
            }

            H_mat(idx_ss, idx_ss) += w_ss;
            b(idx_ss) += w_ss * (scale - prior_s);
            total_cost += 0.5 * w_ss * (scale - prior_s) * (scale - prior_s);

            // Shift prior (when optimized)
            if (opt_shift) {
                double prior_h = init_scaleshift ? init_scaleshift[1] : 0.0;
                H_mat(idx_ss+1, idx_ss+1) += w_ss;
                b(idx_ss+1) += w_ss * (shift - prior_h);
                total_cost += 0.5 * w_ss * (shift - prior_h) * (shift - prior_h);
            }
        }
 
        // --- F. Solve & Update ---
        if (iter > 0 && total_cost >= last_cost) {
            // Rollback
            t_cur = t_best; R_cur = R_best;
            alpha = alpha_best; beta = beta_best;
            scale = scale_best; shift = shift_best;
            break;
        }
        
        // Save best
        last_cost = total_cost;
        t_best = t_cur; R_best = R_cur;
        alpha_best = alpha; beta_best = beta;
        scale_best = scale; shift_best = shift;
        H_best = H_mat;
        
        // Solve Hx = -b
        // LM Damping to handle ill-conditioning (huge dx)
        Eigen::MatrixXd H_damped = H_mat.topLeftCorner(dim, dim);
        double lambda = 1e-4; 
        H_damped.diagonal().array() *= (1.0 + lambda);
        H_damped.diagonal().array() += 1e-6; // Absolute damping floor

        auto solver = H_damped.ldlt();
        Eigen::VectorXd dx = solver.solve(-b.head(dim));
        
        if (solver.info() != Eigen::Success) {
            std::ostringstream ss;
            ss << "[GN-EXT ERROR] LDLT Solver Failed! Info=" << solver.info() << "\n";
            std::cout << ss.str();
            break;
        }

        if ((dx.array() != dx.array()).any()) {
            std::cout << "[GN-EXT ERROR] NaN in dx!" << std::endl;
            break; 
        }

        // Update Pose
        Eigen::Vector3d dt = dx.head<3>();
        Eigen::Vector3d dw = dx.segment<3>(3);
        
        // Update Affine
        if (opt_affine) {
            alpha += dx(idx_aff);
            beta  += dx(idx_aff + 1);
        }
        
        // Update Scale (+Shift if enabled)
        if (opt_scaleshift) {
            scale += dx(idx_ss);
            if (opt_shift) {
                shift += dx(idx_ss + 1);
            }
        }

        // Apply Pose Update (Left Perturbation)
        double theta = dw.norm();
        Eigen::Matrix3d dR;
        if (theta < 1e-10) dR.setIdentity();
        else dR = Eigen::AngleAxisd(theta, dw/theta).toRotationMatrix();
        
        t_cur = dR * t_cur + dt;
        R_cur = dR * R_cur;

        // Check Convergence
        if (dx.norm() < 1e-6) break;
    }


    // Fill Result
    result.q_cur_ref = Eigen::Quaterniond(R_cur).normalized();
    result.t_cur_ref = t_cur;
    result.alpha = alpha;
    result.beta = beta;
    result.scale = scale;
    result.shift = shift;
    result.final_cost = last_cost;
    result.num_valid = final_num_valid;
    
    // Store full Hessian (padded with zeros if not used, or just top-left)
    // We store whatever we computed, users should check 'dim' or flags.
    // Ideally we assume the user knows the layout based on flags.
    // You might want to clear the unused parts of H_full to be safe.
    // Store full Hessian
    result.H_full = H_best;
    
    //printf("[GN-EXT DEBUG] Final: Cost=%.4e, Aff=(%.4f,%.4f), SS=(%.4f,%.4f), Iters=%d\n", 
    //       last_cost, alpha, beta, scale, shift, result.iterations);
    
    return result;
}

void PhotometricRefinement::threadLoop0() {
    while (keep_running_) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(task_mutex_0_);
            task_cond_0_.wait(lock, [this] { return !task_queue_0_.empty() || !keep_running_; });
            
            if (!keep_running_) break;
            
            task = task_queue_0_.front();
            task_queue_0_.pop();
        }
        
        // Set busy flag
        thread_0_busy_.store(true);

        // --- PERFORM OPTIMIZATION ---
        auto t_start = std::chrono::high_resolution_clock::now();
        
        double pose_cur_ref[7];
        pose_cur_ref[0] = task.t_initial.x();
        pose_cur_ref[1] = task.t_initial.y();
        pose_cur_ref[2] = task.t_initial.z();
        pose_cur_ref[3] = task.q_initial.x();
        pose_cur_ref[4] = task.q_initial.y();
        pose_cur_ref[5] = task.q_initial.z();
        pose_cur_ref[6] = task.q_initial.w();

        // Dummy Identity pose for the reference frame (we only optimize the relative transform)
        // PhotometricRegFactor takes pose_src and pose_tgt and computes T_tgt_src.
        // Here src=Ref, tgt=Cur.
        // We set pose_src to Identity and optimize pose_tgt.
        // T_tgt_src = T_tgt^{-1} * T_src = T_cur^{-1} * I = T_cur^{-1}?
        // Wait, let's check PhotometricRegFactor logic:
        // T_relative = T_w_tgt^-1 * T_w_src
        // If we want T_relative to be T_cur_ref (Ref to Cur), then T_cur_ref = T_cur^{-1} ? No.
        // T_cur_ref means P_cur = T_cur_ref * P_ref.
        // Conventional VINS T_w_i. P_w = T_w_i * P_i.
        // P_cur = T_w_cur^{-1} * P_w = T_w_cur^{-1} * T_w_ref * P_ref.
        // So T_cur_ref = T_w_cur^{-1} * T_w_ref.
        
        // In PhotometricRegFactor:
        // T_relative = R_tgt^T * R_src ... this is indeed T_cur_ref if pose_tgt=T_w_cur and pose_src=T_w_ref.
        
        // So:
        // pose_src (Ref) = Identity (Origin)
        // pose_tgt (Cur) = T_ref_cur (Cur w.r.t Ref) ? No.
        // If pose_src = I, then T_w_src = I.
        // T_relative = T_w_tgt^{-1} * I = T_w_tgt^{-1}.
        // This is T_tgt_w (from World(Ref) to Target).
        // This is exactly T_cur_ref (Ref is World).
        
        // So if we set pose_src=Identity and optimize pose_tgt, the resulting pose_tgt is T_ref_cur (pose of Cur in Ref frame).
        // And T_relative computed inside factor will be T_cur_ref.
        
        // WAIT! Standard VINS pose is T_w_b (Body to World).
        // pose_tgt IS T_ref_cur (Position of Cur in Ref frame).
        // Then T_w_tgt (if world is Ref) is T_ref_cur.
        // Factor computes T_relative = T_w_tgt^{-1} * T_w_src
        // = T_ref_cur^{-1} * I = T_cur_ref.
        // This checks out.
        
        // So:
        // Parameter Block 0 (Ref): Fixed at Identity.
        // Parameter Block 1 (Cur): Optimized, initialized with T_ref_cur? NO.
        // The factor takes global poses.
        // If we want to optimize relative pose T_cur_ref directly?
        // We act AS IF 'Ref' is at global origin.
        // Then 'Cur' is at T_ref_cur (inverse of T_cur_ref).
        // Let's optimize T_ref_cur (Pose of Cur in Ref).
        
        // Initial guess for T_ref_cur:
        // task.q_initial / t_initial are T_cur_ref (Ref -> Cur transform).
        // Sparse Factor expects T_cur_ref (Point Ref -> Point Cur).
        // Dense Factor expects T_ref_cur (Pose of Cur in Ref) and internally inverts it.
        
        Eigen::Quaterniond q_cur_ref = task.q_initial;
        Eigen::Vector3d t_cur_ref = task.t_initial;
        
        // Default to T_ref_cur (Dense Mode Logic)
        Eigen::Quaterniond q_init_opt = q_cur_ref.inverse();
        Eigen::Vector3d t_init_opt = -(q_init_opt * t_cur_ref);
        
        if (use_sparse_) {
             // Sparse Mode: Optimize T_cur_ref directly
             q_init_opt = q_cur_ref;
             t_init_opt = t_cur_ref;
        }

        double p_ref[7] = {0,0,0, 0,0,0,1}; // Identity
        // Initialize p_cur with the correct transform for the mode
        Eigen::Vector3d t_pert = t_init_opt;
        Eigen::Quaterniond q_pert = q_init_opt;
        
       

        double p_cur[7] = {t_pert.x(), t_pert.y(), t_pert.z(), 
                           q_pert.x(), q_pert.y(), q_pert.z(), q_pert.w()};
                           
        // GN tracking variables (replaces ceres::Solver::Summary)
        int num_factors = 0;
        double gn_initial_cost = 0, gn_final_cost = 0;
        int gn_iterations = 0;
        Eigen::Matrix<double, 6, 6> gn_hessian = Eigen::Matrix<double, 6, 6>::Zero();
        double affine[2] = {1.0, 0.0};
        double scale_shift[2] = {task.init_scale, task.init_shift};

        ceres::Solver::Summary summary; // Only used for dense mode fallback

        if (use_sparse_) {
            // --- COARSE-TO-FINE PYRAMID (DSO-style) ---
            //PYRAMID
            const int NUM_LEVELS = 1; // Enable pyramid for large basin of attraction
            printf("[Pyramid] Building %d-level pyramid. img_ref: %dx%d, depth: %dx%d (type=%d, empty=%d)\n",
                   NUM_LEVELS, task.img_ref.cols, task.img_ref.rows,
                   task.depth_ref.cols, task.depth_ref.rows, task.depth_ref.type(), task.depth_ref.empty());

            // Build Image Pyramids
            std::vector<cv::Mat> pyr_ref(NUM_LEVELS), pyr_cur(NUM_LEVELS), pyr_depth(NUM_LEVELS);
            std::vector<Eigen::Matrix3d> pyr_K(NUM_LEVELS);

            pyr_ref[0] = task.img_ref;
            pyr_cur[0] = task.img_cur;
            pyr_depth[0] = task.depth_ref;
            pyr_K[0] = task.K;

            for (int l = 1; l < NUM_LEVELS; l++) {
                cv::pyrDown(pyr_ref[l-1], pyr_ref[l]);
                cv::pyrDown(pyr_cur[l-1], pyr_cur[l]);
                // Depth: use INTER_NEAREST to avoid blending inverse depth values
                cv::Mat depth_down;
                cv::resize(pyr_depth[l-1], depth_down,
                          cv::Size(pyr_depth[l-1].cols/2, pyr_depth[l-1].rows/2),
                          0, 0, cv::INTER_NEAREST);
                pyr_depth[l] = depth_down;

                // Scale K: halve fx, fy, cx, cy
                pyr_K[l] = pyr_K[l-1];
                pyr_K[l](0,0) *= 0.5; // fx
                pyr_K[l](1,1) *= 0.5; // fy
                pyr_K[l](0,2) *= 0.5; // cx
                pyr_K[l](1,2) *= 0.5; // cy
                printf("[Pyramid] Level %d: %dx%d, depth %dx%d\n", l, pyr_ref[l].cols, pyr_ref[l].rows, pyr_depth[l].cols, pyr_depth[l].rows);
            }

            // Iterate from COARSEST to FINEST
            for (int level = NUM_LEVELS - 1; level >= 0; level--) {
                printf("[Pyramid] Starting level %d\n", level);
                const cv::Mat& img_ref_l = pyr_ref[level];
                const cv::Mat& img_cur_l = pyr_cur[level];
                const cv::Mat& depth_l = pyr_depth[level];
                const Eigen::Matrix3d& K_l = pyr_K[level];

                // 1. Select Features (with pattern expansion)
                std::vector<FeaturePoint> features;
                // Reduced grid: fewer centers, each with ~8 pattern members
                // ~800 features × ~8 pattern = ~6400 residuals (vs old 4096 point-wise)
                int grid = (level == 0) ? 24 : 14;
                int per_cell = (level == 0) ? 2 : 1;
                selectPixelFeatures(img_ref_l, depth_l, features, grid, per_cell);

                if (features.size() < 10) {
                    printf("[Refinement] Level %d: too few features (%zu), skipping\n", level, features.size());
                    continue;
                }

                // --- GAUSS-NEWTON PHOTOMETRIC OPTIMIZATION ---
                Eigen::Quaterniond q_level(p_cur[6], p_cur[3], p_cur[4], p_cur[5]);
                Eigen::Vector3d t_level(p_cur[0], p_cur[1], p_cur[2]);
                bool do_affine = (mode_ == POSE_AFFINE || mode_ == POSE_FULL || mode_ == POSE_FULL_SHIFT);
                bool do_ss     = (mode_ == POSE_SCALE_SHIFT || mode_ == POSE_FULL || mode_ == POSE_FULL_SHIFT);
                bool do_shift  = (mode_ == POSE_FULL_SHIFT);

                double init_aff[2] = { affine[0], affine[1] };
                double init_ss[2]  = { scale_shift[0], scale_shift[1] };
                auto gn = solvePhotometricGN_Extended(
                    features, img_cur_l, K_l,
                    q_level, t_level,
                    50,
                    do_affine,
                    do_ss,
                    init_aff,
                    init_ss,
                    do_shift
                );

                p_cur[0] = gn.t_cur_ref.x();
                p_cur[1] = gn.t_cur_ref.y();
                p_cur[2] = gn.t_cur_ref.z();
                p_cur[3] = gn.q_cur_ref.x();
                p_cur[4] = gn.q_cur_ref.y();
                p_cur[5] = gn.q_cur_ref.z();
                p_cur[6] = gn.q_cur_ref.w();
                if (do_affine) {
                    affine[0] = gn.alpha;
                    affine[1] = gn.beta;
                }
                if (do_ss) {
                    scale_shift[0] = gn.scale;
                    scale_shift[1] = gn.shift;
                }

                num_factors = gn.num_valid;
                gn_initial_cost = gn.initial_cost;
                gn_final_cost = gn.final_cost;
                gn_iterations = gn.iterations;
                gn_hessian = gn.H_full.topLeftCorner<6,6>();
            }

            // --- Dense Depth-to-Depth Scale/Shift Refinement ---
            if (!task.depth_cur.empty() &&
                (mode_ == POSE_SCALE_SHIFT || mode_ == POSE_FULL || mode_ == POSE_FULL_SHIFT)) {
                Eigen::Quaterniond q_refined(p_cur[6], p_cur[3], p_cur[4], p_cur[5]);
                Eigen::Matrix3d R_refined = q_refined.toRotationMatrix();
                Eigen::Vector3d t_refined(p_cur[0], p_cur[1], p_cur[2]);

                auto ss_result = refineDenseScaleShift(
                    task.depth_ref, task.depth_cur,
                    R_refined, t_refined, task.K,
                    scale_shift[0], scale_shift[1],
                    task.init_scale_cur, task.init_shift_cur);

                if (ss_result.num_valid >= 100) {
                    printf("[DenseSS] Refined: s=%.4f->%.4f, h=%.6f->%.6f\n",
                           scale_shift[0], ss_result.scale, scale_shift[1], ss_result.shift);
                    scale_shift[0] = ss_result.scale;
                    scale_shift[1] = ss_result.shift;
                }
            }

        } else {
            // --- DENSE MODE (Legacy, single level) ---
            ceres::Problem problem;
            ceres::LossFunction* loss_function = new ceres::HuberLoss(huber_loss_);

            switch(mode_) {
                case POSE_ONLY: {
                    ceres::CostFunction* cost_function = PhotometricRegFactor::Create(
                        task.img_ref, task.img_cur, task.depth_ref, task.K,
                        1.0, ssim_weight_, l1_weight_);
                    problem.AddResidualBlock(cost_function, loss_function, p_ref, p_cur);
                    break;
                }

                case POSE_AFFINE: {
                    ceres::CostFunction* cost_function_affine = PhotometricRegFactorAffine::Create(
                        task.img_ref, task.img_cur, task.depth_ref, task.K,
                        1.0, ssim_weight_, l1_weight_);
                    problem.AddResidualBlock(cost_function_affine, loss_function, p_ref, p_cur, affine);
                    break;
                }

                case POSE_SCALE_SHIFT: {
                    ceres::CostFunction* cost_function_ss = PhotometricRegFactorScaleShift::Create(
                        task.img_ref, task.img_cur, task.depth_ref, task.K,
                        1.0, ssim_weight_, l1_weight_);
                    problem.AddResidualBlock(cost_function_ss, loss_function, p_ref, p_cur, scale_shift);
                    break;
                }

                default: {
                    ceres::CostFunction* cost_function = PhotometricRegFactor::Create(
                        task.img_ref, task.img_cur, task.depth_ref, task.K,
                        1.0, ssim_weight_, l1_weight_);
                    problem.AddResidualBlock(cost_function, loss_function, p_ref, p_cur);
                    break;
                }
            }

            // Fix Ref pose
            if (problem.HasParameterBlock(p_ref)) {
                 problem.SetParameterBlockConstant(p_ref);
            }
            if (mode_ == POSE_SCALE_SHIFT) {
                 if (problem.HasParameterBlock(p_cur)) {
                    problem.SetParameterBlockConstant(p_cur);
                 }
            }

            auto* manifold = new ceres::ProductManifold<
                ceres::EuclideanManifold<3>,
                ceres::EigenQuaternionManifold>();
            problem.SetManifold(p_cur, manifold);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 30;
            options.minimizer_progress_to_stdout = false;
            options.num_threads = 1;

            ceres::Solve(options, &problem, &summary);
        }

        // Extract result
        RefinementResult result;
        result.timestamp_ref = task.t_ref;
        result.timestamp_cur = task.t_cur;

        // Unify cost variables: use GN for sparse, ceres summary for dense
        double opt_initial_cost = use_sparse_ ? gn_initial_cost : summary.initial_cost;
        double opt_final_cost = use_sparse_ ? gn_final_cost : summary.final_cost;
        int opt_iterations = use_sparse_ ? gn_iterations : (int)summary.iterations.size();
        bool opt_success = use_sparse_ ? (gn_final_cost < gn_initial_cost || gn_iterations > 0)
                                       : (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < summary.initial_cost);

        if (opt_success) {
            // Extract result
            Eigen::Vector3d t_res(p_cur[0], p_cur[1], p_cur[2]);
            Eigen::Quaterniond q_res(p_cur[6], p_cur[3], p_cur[4], p_cur[5]);
            if (use_sparse_) {
                 // We optimized T_cur_ref. Result expected is T_ref_cur.
                 // Invert back.
                 Eigen::Quaterniond q_inv = q_res.inverse();
                 Eigen::Vector3d t_inv = -(q_inv * t_res);
                 result.t_ref_cur = t_inv;
                 result.q_ref_cur = q_inv;
            } else {
                 // We optimized T_ref_cur directly.
                 result.t_ref_cur = t_res;
                 result.q_ref_cur = q_res;
            }

            // Standardize Quaternion Sign (w > 0)
            if (result.q_ref_cur.w() < 0) {
                result.q_ref_cur.coeffs() = -result.q_ref_cur.coeffs();
            }

            result.alpha = affine[0];
            result.beta = affine[1];
            result.scale = scale_shift[0];
            result.shift = scale_shift[1];
            result.feature_count = num_factors;
            result.initial_cost = gn_initial_cost;
            result.final_cost = gn_final_cost;
            result.success = true;

            // Calculate Hessian / Information Matrix
            if (use_sparse_) {
                // Single global scale preserving eigenstructure (Z-axis naturally gets lower weight)
                double trace_full = gn_hessian.trace();
                double global_scale = (trace_full > 0) ? 6000.0 / trace_full : 1000.0;
                result.information = gn_hessian * global_scale;
                printf("[Refinement] Hessian trace: %.4e, global_scale: %.4e\n", trace_full, global_scale);
                std::cout << "[Refinement] Information matrix diag: "
                          << result.information.diagonal().transpose().format(Eigen::IOFormat(4)) << std::endl;
            } else {
                // Dense mode: we have a Problem object
                ceres::Covariance::Options cov_options;
                ceres::Covariance covariance(cov_options);
                
                std::vector<std::pair<const double*, const double*>> covariance_blocks;
                
                if (mode_ == POSE_SCALE_SHIFT) {
                    covariance_blocks.push_back(std::make_pair(scale_shift, scale_shift));
                } else {
                    covariance_blocks.push_back(std::make_pair(p_cur, p_cur));
                }
                
                // Dense mode fallback weight
                result.information = Eigen::Matrix<double, 6, 6>::Identity() * 2.0e4;
            }
            // printf("[Refinement] Success. Mode %d. Cost: %.4f -> %.4f\n", mode_, summary.initial_cost, summary.final_cost);
            
            // --- DEBUG PRINTS REQUESTED BY USER ---
            // 1. Pose Comparison
            // VINS Pose (Initial Guess)
            Eigen::Vector3d t_vins = task.t_initial;
            Eigen::Quaterniond q_vins = task.q_initial; // T_cur_ref
            
            // Refined Pose (for debug we want T_cur_ref)
            Eigen::Quaterniond q_opt_cur_ref;
            Eigen::Vector3d t_opt_cur_ref;
            
            if (use_sparse_) {
                // p_cur was T_cur_ref
                 t_opt_cur_ref = t_res;
                 q_opt_cur_ref = q_res;
            } else {
                // p_cur was T_ref_cur, so invert for debug
                 q_opt_cur_ref = q_res.inverse();
                 t_opt_cur_ref = -(q_opt_cur_ref * t_res);
            }
            
            // Re-evaluate to get valid pixel count
            /*
            cv::Mat map_x, map_y, valid_mask;
            PhotometricLoss::generateWarpMaps(
                task.depth_ref, task.K, 
                // T_cur_ref (Points Ref -> Cur)
                // We need T_target_src.
                // src=Ref, tgt=Cur.
                // T_cur_ref matches.
                // We need T_cur_ref (Pose of Ref in Cur frame? No. Points Ref to Cur).
                // P_cur = T * P_ref.
                // T_cur_ref IS this transform.
                // t_res_cur_ref, q_res_cur_ref.
                Eigen::Affine3d(q_res_cur_ref) * Eigen::Translation3d(t_res_cur_ref),
                map_x, map_y, valid_mask, 
                result.scale, result.shift
            );
            int valid_pixels = cv::countNonZero(valid_mask);
            */
            // Skipping expensive pixel count for now to avoid latency.
            int valid_pixels = -1; 
            
            auto t_end = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            
            printf("\n[Photometric Refinement] Mode: %d | Frame: %.3f -> %.3f | Time: %.2f ms\n", mode_, task.t_ref, task.t_cur, duration_ms);
            printf("  Iterations: %d | Cost: %.4e -> %.4e\n", opt_iterations, opt_initial_cost, opt_final_cost);
            printf("  VINS Pose (T_cur_ref): t=[%.3f, %.3f, %.3f] q=[%.3f, %.3f, %.3f, %.3f]\n",
                   t_vins.x(), t_vins.y(), t_vins.z(), q_vins.w(), q_vins.x(), q_vins.y(), q_vins.z());
            printf("  Ours Pose (T_cur_ref): t=[%.3f, %.3f, %.3f] q=[%.3f, %.3f, %.3f, %.3f]\n",
                   t_opt_cur_ref.x(), t_opt_cur_ref.y(), t_opt_cur_ref.z(), 
                   q_opt_cur_ref.w(), q_opt_cur_ref.x(), q_opt_cur_ref.y(), q_opt_cur_ref.z());
            printf("mode_=%d\n", mode_);
            // --- ADD THIS BLOCK ---
            if (mode_ == POSE_AFFINE || mode_ == POSE_FULL || mode_ == POSE_FULL_SHIFT) {
                printf("  Affine: alpha=%.5f, beta=%.5f\n", result.alpha, result.beta);
            }
            if (mode_ == POSE_SCALE_SHIFT || mode_ == POSE_FULL || mode_ == POSE_FULL_SHIFT) {
                printf("  Scale/Shift: scale=%.5f, shift=%.5f\n", result.scale, result.shift);
            }
            // ----------------------
            // High-precision delta diagnostic
            Eigen::Vector3d dt = t_opt_cur_ref - t_vins;
            Eigen::Quaterniond dq = q_opt_cur_ref * q_vins.inverse();
            Eigen::AngleAxisd aa(dq);
            double angle_deg = aa.angle() * 180.0 / M_PI;
            printf("  DELTA: dt=[%.6f, %.6f, %.6f] mm  |dt|=%.4f mm  angle=%.4f deg\n",
                   dt.x()*1000, dt.y()*1000, dt.z()*1000, dt.norm()*1000, angle_deg);
            printf("  Cost reduction: %.2f%%\n", 
                   (opt_initial_cost > 0) ? 100.0 * (1.0 - opt_final_cost / opt_initial_cost) : 0.0);
            
        } else {
            result.success = false;
            printf("[Photometric Refinement] FAILED due to convergence issues. Cost: %.4e -> %.4e\n", opt_initial_cost, opt_final_cost);
        }

        // --- KAIZEN: Guard Rails / Gating ---
        if (result.success) {
            double dt = std::abs(result.timestamp_cur - result.timestamp_ref);
            if (dt < 1e-4) dt = 1e-4; // Avoid div by zero
            
            double v_est = result.t_ref_cur.norm() / dt;
            
            Eigen::AngleAxisd aa(result.q_ref_cur);
            double rot_angle = aa.angle();
            
            // Thresholds: 5 m/s, 0.5 rad (~30 deg)
            bool velocity_check = (v_est < 5.0); 
            bool rotation_check = (std::abs(rot_angle) < 0.5);
            bool cost_check = (opt_final_cost < opt_initial_cost);
            if (!velocity_check) {
                result.success = false;
                printf("\033[1;31m[Photometric Refinement] REJECTED (Velocity): v=%.2f m/s (> 5.0)\033[0m\n", v_est);
            } else if (!rotation_check) {
                result.success = false;
                printf("\033[1;31m[Photometric Refinement] REJECTED (Rotation): angle=%.2f rad (> 0.5)\033[0m\n", rot_angle);
            } else if (!cost_check) {
                 result.success = false;
                 printf("\033[1;31m[Photometric Refinement] REJECTED (Cost): Cost increased or flat\033[0m\n");
            } else {
                 printf("\033[1;32m[Photometric Refinement] ACCEPTED. v=%.2f m/s, angle=%.2f rad\033[0m\n", v_est, rot_angle);
            }
        }
        
        // Push result
        {
            std::lock_guard<std::mutex> lock(result_mutex_0_);
            result_queue_0_.push(result);
        }
        
        // Clear busy flag
        thread_0_busy_.store(false);
    }
}



// ============================================================================
// Dense Depth-to-Depth Scale/Shift Refinement
// After photometric GN refines pose, this uses dense depth maps from both
// frames to directly optimize scale/shift with zero pose coupling.
// Residual (inv depth space): r = 1/P_cur.z - (s_cur * d_cur_raw + h_cur)
// where P_cur = R * (Z_ref * K^{-1}[u,v,1]) + t, Z_ref = 1/(s_ref*d_ref+h_ref)
// ============================================================================
PhotometricRefinement::DenseSSResult PhotometricRefinement::refineDenseScaleShift(
    const cv::Mat& depth_ref, const cv::Mat& depth_cur,
    const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
    const Eigen::Matrix3d& K,
    double s_ref, double h_ref,
    double s_cur, double h_cur,
    int subsample)
{
    DenseSSResult result;
    result.scale = s_ref;
    result.shift = h_ref;
    result.num_valid = 0;
    result.final_cost = 0;

    if (depth_ref.empty() || depth_cur.empty()) return result;

    const double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    const int W_ref = depth_ref.cols, H_ref = depth_ref.rows;
    const int W_cur = depth_cur.cols, H_cur = depth_cur.rows;
    const double huber_thresh = 0.1; // in inverse depth space

    // Precompute R third row for rn = R[2,:] . n
    Eigen::Vector3d R_row2 = R.row(2);
    double tz = t.z();

    const int max_iters = 5;
    double last_cost = std::numeric_limits<double>::max();
    double s_best = s_ref, h_best = h_ref;

    for (int iter = 0; iter < max_iters; iter++) {
        // 2x2 Hessian and 2x1 gradient for (s_ref, h_ref)
        double H00 = 0, H01 = 0, H11 = 0;
        double b0 = 0, b1 = 0;
        double total_cost = 0;
        int num_valid = 0;

        for (int v = subsample/2; v < H_ref; v += subsample) {
            const float* row_ref = depth_ref.ptr<float>(v);
            for (int u = subsample/2; u < W_ref; u += subsample) {
                float d_ref_raw = row_ref[u];
                if (d_ref_raw <= 0.02f || d_ref_raw >= 10.0f) continue;

                // Ref inv depth → metric depth
                double d_ref_fitted = s_ref * d_ref_raw + h_ref;
                if (d_ref_fitted <= 1e-4) continue;
                double Z_ref = 1.0 / d_ref_fitted;

                // Back-project and transform
                double xn = (u - cx) / fx;
                double yn = (v - cy) / fy;
                Eigen::Vector3d n(xn, yn, 1.0);
                double rn = R_row2.dot(n);
                double P_cur_z = Z_ref * rn + tz;
                if (P_cur_z <= 0.01) continue;

                // Project to cur frame
                Eigen::Vector3d P_ref_3d = Z_ref * n;
                Eigen::Vector3d P_cur_3d = R * P_ref_3d + t;
                double u_proj = fx * P_cur_3d.x() / P_cur_z + cx;
                double v_proj = fy * P_cur_3d.y() / P_cur_z + cy;

                if (u_proj < 0 || u_proj >= W_cur - 1 || v_proj < 0 || v_proj >= H_cur - 1) continue;

                // Bilinear interpolate target depth
                int u0 = (int)u_proj, v0 = (int)v_proj;
                double du = u_proj - u0, dv = v_proj - v0;
                double w00 = (1-du)*(1-dv), w10 = du*(1-dv);
                double w01 = (1-du)*dv,     w11 = du*dv;
                const float* cr0 = depth_cur.ptr<float>(v0);
                const float* cr1 = depth_cur.ptr<float>(v0+1);
                double d_cur_raw = w00*cr0[u0] + w10*cr0[u0+1] + w01*cr1[u0] + w11*cr1[u0+1];

                if (d_cur_raw <= 0.02 || d_cur_raw >= 10.0) continue;

                // Residual: projected inv depth vs target fitted inv depth
                double inv_z_proj = 1.0 / P_cur_z;
                double inv_z_obs = s_cur * d_cur_raw + h_cur;
                if (inv_z_obs <= 1e-4) continue;

                double r = inv_z_proj - inv_z_obs;

                // Huber weight
                double abs_r = std::abs(r);
                double w = (abs_r > huber_thresh) ? (huber_thresh / abs_r) : 1.0;

                // Jacobian of residual w.r.t. (s_ref, h_ref)
                // inv_z_proj = 1/P_cur_z, P_cur_z = Z_ref*rn + tz, Z_ref = 1/(s*d+h)
                // d(inv_z_proj)/dZ_ref = -rn / P_cur_z^2
                // dZ_ref/ds = -Z_ref^2 * d_ref_raw
                // dZ_ref/dh = -Z_ref^2
                double Z_ref_sq = Z_ref * Z_ref;
                double P_z_sq = P_cur_z * P_cur_z;
                double common = -rn * Z_ref_sq / P_z_sq;  // d(inv_z)/dZ_ref * dZ_ref/d(...)
                double J_s = common * (-d_ref_raw);  // w.r.t. scale: chain with dZ/ds = -Z^2*d
                double J_h = common * (-1.0);        // w.r.t. shift: chain with dZ/dh = -Z^2

                // Wait, let me redo: d(1/P_z)/ds = d(1/P_z)/dP_z * dP_z/dZ_ref * dZ_ref/ds
                // d(1/P_z)/dP_z = -1/P_z^2
                // dP_z/dZ_ref = rn
                // dZ_ref/ds = -Z_ref^2 * d_raw
                // So: J_s = (-1/P_z^2) * rn * (-Z_ref^2 * d_raw) = rn * Z_ref^2 * d_raw / P_z^2
                J_s = rn * Z_ref_sq * d_ref_raw / P_z_sq;
                J_h = rn * Z_ref_sq / P_z_sq;

                // Accumulate 2x2 system
                H00 += w * J_s * J_s;
                H01 += w * J_s * J_h;
                H11 += w * J_h * J_h;
                b0  += w * J_s * r;
                b1  += w * J_h * r;

                if (abs_r <= huber_thresh) total_cost += 0.5 * r * r;
                else total_cost += huber_thresh * (abs_r - 0.5 * huber_thresh);

                num_valid++;
            }
        }

        if (num_valid < 100) {
            printf("[DenseSS] Too few valid pixels (%d), aborting\n", num_valid);
            break;
        }

        if (iter > 0 && total_cost >= last_cost) {
            s_ref = s_best; h_ref = h_best;
            break;
        }
        last_cost = total_cost;
        s_best = s_ref; h_best = h_ref;

        // Solve 2x2: [H00 H01; H01 H11] * [ds; dh] = -[b0; b1]
        double det = H00 * H11 - H01 * H01;
        if (std::abs(det) < 1e-12) break;
        double ds = -(H11 * b0 - H01 * b1) / det;
        double dh = -(H00 * b1 - H01 * b0) / det;

        s_ref += ds;
        h_ref += dh;

        if (iter == 0) {
            printf("[DenseSS] iter0: valid=%d, cost=%.4e, s=%.4f, h=%.6f\n",
                   num_valid, total_cost, s_ref, h_ref);
        }

        if (std::abs(ds) < 1e-6 && std::abs(dh) < 1e-6) break;

        result.num_valid = num_valid;
    }

    result.scale = s_ref;
    result.shift = h_ref;
    result.final_cost = last_cost;

    printf("[DenseSS] Final: s=%.4f h=%.6f (valid=%d, cost=%.4e)\n",
           result.scale, result.shift, result.num_valid, result.final_cost);

    return result;
}

void PhotometricRefinement::selectPixelFeatures(const cv::Mat& img, const cv::Mat& depth, 
                                                std::vector<FeaturePoint>& features,
                                                int grid_size, int features_per_grid) {
    // Image gradients
    cv::Mat dx, dy;
    cv::Sobel(img, dx, CV_16S, 1, 0, 3);
    cv::Sobel(img, dy, CV_16S, 0, 1, 3);

    // Depth gradient magnitude (pre-computed once, O(N) — avoids per-candidate 3x3 patch)
    // Soft penalty: features on depth edges get lower score instead of hard rejection
    cv::Mat depth_dx, depth_dy, depth_grad_sq;
    cv::Sobel(depth, depth_dx, CV_32F, 1, 0, 3);
    cv::Sobel(depth, depth_dy, CV_32F, 0, 1, 3);
    cv::multiply(depth_dx, depth_dx, depth_dx);
    cv::multiply(depth_dy, depth_dy, depth_dy);
    depth_grad_sq = depth_dx + depth_dy;
    // Threshold: above this, features are on depth discontinuities
    // Tuned for inverse depth maps where edge gradients are >> interior gradients
    const float depth_edge_threshold = 0.01f;

    int w = img.cols;
    int h = img.rows;

    int cell_w = w / grid_size;
    int cell_h = h / grid_size;

    double grad_threshold_sq = 20.0 * 20.0;

    // Periphery boost for Z-axis observability
    double cx = w * 0.5, cy = h * 0.5;
    double max_radius = std::hypot(cx, cy);

    features.clear();
    features.reserve(grid_size * grid_size * features_per_grid);

    // Grid Selection
    for (int gy = 0; gy < grid_size; gy++) {
        for (int gx = 0; gx < grid_size; gx++) {
            int x0 = gx * cell_w;
            int y0 = gy * cell_h;
            int x1 = std::min(x0 + cell_w, w - 2);
            int y1 = std::min(y0 + cell_h, h - 2);

            struct Candidate { int x, y; double score; };
            std::vector<Candidate> candidates;

            for (int v = y0 + 1; v < y1; v++) {
                const short* row_dx = dx.ptr<short>(v);
                const short* row_dy = dy.ptr<short>(v);
                const float* row_dg = depth_grad_sq.ptr<float>(v);

                for (int u = x0 + 1; u < x1; u++) {
                    double g_sq = row_dx[u]*row_dx[u] + row_dy[u]*row_dy[u];
                    if (g_sq > grad_threshold_sq) {
                        float d = depth.at<float>(v, u);
                        if (d > 0.02 && d < 10.0) {
                            // Soft depth-edge penalty: 1.0 in smooth regions, ~0 on depth edges
                            double depth_penalty = 1.0 / (1.0 + row_dg[u] / depth_edge_threshold);

                            // Periphery boost (1x center → 3x edge)
                            double dist = std::hypot(u - cx, v - cy) / max_radius;
                            double peripheral_boost = 1.0 + 2.0 * dist;

                            double score = g_sq * peripheral_boost * depth_penalty;
                            candidates.push_back({u, v, score});
                        }
                    }
                }
            }

            // Sort by boosted score (descending) and pick top N
            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
            
            int n_pick = std::min(features_per_grid, (int)candidates.size());
            for (int k = 0; k < n_pick; k++) {
                 FeaturePoint fp;
                 fp.u = Eigen::Vector2d(candidates[k].x, candidates[k].y);
                 fp.intensity = (double)img.at<uchar>(candidates[k].y, candidates[k].x);
                 fp.inv_depth = depth.at<float>(candidates[k].y, candidates[k].x);
                 fp.pattern_count = 0;
                 features.push_back(fp);
            }
        }
    }

    // --- Pattern expansion: depth-validated sparse samples around each center ---
    // DSO-inspired spread pattern (not a dense block — samples wider area)
    static const int PATTERN_OFFSETS[8][2] = {
        {0, -2}, {-1, -1}, { 2,  0}, { 1, -1},
        {0,  2}, { 1,  1}, {-2,  0}, {-1,  1}
    };
    const float depth_consistency_thresh = 0.10f; // 10% relative difference

    for (auto& fp : features) {
        int uc = (int)fp.u.x(), vc = (int)fp.u.y();
        float d_center = fp.inv_depth;
        fp.pattern_count = 0;

        for (int p = 0; p < 8; p++) {
            int nu = uc + PATTERN_OFFSETS[p][0];
            int nv = vc + PATTERN_OFFSETS[p][1];

            if (nu < 1 || nu >= w - 1 || nv < 1 || nv >= h - 1) continue;

            float d_neighbor = depth.at<float>(nv, nu);
            if (d_neighbor <= 0.02f || d_neighbor >= 10.0f) continue;

            // Depth consistency: reject if on depth boundary
            if (std::abs(d_neighbor - d_center) / d_center > depth_consistency_thresh) continue;

            fp.pattern[fp.pattern_count].du = PATTERN_OFFSETS[p][0];
            fp.pattern[fp.pattern_count].dv = PATTERN_OFFSETS[p][1];
            fp.pattern[fp.pattern_count].intensity = (double)img.at<uchar>(nv, nu);
            fp.pattern_count++;
        }
    }
}
void PhotometricRefinement::selectPixelFeaturesByDepthPercentile(
    const cv::Mat& img, const cv::Mat& depth,
    std::vector<FeaturePoint>& features,
    double scale, double shift,
    double lower_percentile, double upper_percentile) {
    
    features.clear();
    
    if (depth.empty() || img.empty()) {
        printf("[DepthPercentile] WARNING: Empty depth or image\n");
        return;
    }
    
    int w = img.cols;
    int h = img.rows;
    
    // Step 1: Collect all valid fitted inverse depths
    std::vector<double> fitted_depths;
    fitted_depths.reserve(w * h);
    
    for (int v = 1; v < h - 1; v++) {  // Margin 1px
        for (int u = 1; u < w - 1; u++) {
            float d = depth.at<float>(v, u); // Raw inverse depth
            if (d > 0.02 && d < 10.0) {  // Valid range
                double fitted_d = scale * d + shift;
                fitted_depths.push_back(fitted_d);
            }
        }
    }
    
    if (fitted_depths.size() < 100) {
        printf("[DepthPercentile] WARNING: Too few valid depths (%zu)\n", fitted_depths.size());
        return;
    }
    
    // Step 2: Compute percentiles
    std::sort(fitted_depths.begin(), fitted_depths.end());
    
    size_t idx_lower = (size_t)(lower_percentile * fitted_depths.size());
    size_t idx_upper = (size_t)(upper_percentile * fitted_depths.size());
    
    double depth_min = fitted_depths[idx_lower];
    double depth_max = fitted_depths[idx_upper];
    
    printf("[DepthPercentile] Fitted depth range [%.1f%% - %.1f%%]: [%.4f, %.4f]\n",
           lower_percentile * 100, upper_percentile * 100, depth_min, depth_max);
    
    // Step 3: Select pixels in this depth range with stride-based subsampling
    // Target ~2000 features to keep optimization fast
    int total_in_range = idx_upper - idx_lower;
    int target_features = 2000;
    int stride = std::max(1, (int)std::sqrt((double)total_in_range / target_features));
    
    for (int v = 1; v < h - 1; v += stride) {
        for (int u = 1; u < w - 1; u += stride) {
            float d = depth.at<float>(v, u);
            if (d > 0.02 && d < 10.0) {
                double fitted_d = scale * d + shift;
                
                if (fitted_d >= depth_min && fitted_d <= depth_max) {
                    FeaturePoint fp;
                    fp.u = Eigen::Vector2d(u, v);
                    fp.intensity = (double)img.at<uchar>(v, u);
                    fp.inv_depth = d;  // Store raw inverse depth
                    features.push_back(fp);
                }
            }
        }
    }
    
    printf("[DepthPercentile] Selected %zu features (stride=%d)\n", features.size(), stride);
}

void PhotometricRefinement::threadLoop1() {
    while (keep_running_) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(task_mutex_1_);
            task_cond_1_.wait(lock, [this] { return !task_queue_1_.empty() || !keep_running_; });
            
            if (!keep_running_) break;
            
            task = task_queue_1_.front();
            task_queue_1_.pop();
        }
        
        // Set busy flag
        thread_1_busy_.store(true);

        // --- PERFORM OPTIMIZATION ---
        auto t_start = std::chrono::high_resolution_clock::now();
        
        double pose_cur_ref[7];
        pose_cur_ref[0] = task.t_initial.x();
        pose_cur_ref[1] = task.t_initial.y();
        pose_cur_ref[2] = task.t_initial.z();
        pose_cur_ref[3] = task.q_initial.x();
        pose_cur_ref[4] = task.q_initial.y();
        pose_cur_ref[5] = task.q_initial.z();
        pose_cur_ref[6] = task.q_initial.w();

        // Dummy Identity pose for the reference frame (we only optimize the relative transform)
        // PhotometricRegFactor takes pose_src and pose_tgt and computes T_tgt_src.
        // Here src=Ref, tgt=Cur.
        // We set pose_src to Identity and optimize pose_tgt.
        // T_tgt_src = T_tgt^{-1} * T_src = T_cur^{-1} * I = T_cur^{-1}?
        // Wait, let's check PhotometricRegFactor logic:
        // T_relative = T_w_tgt^-1 * T_w_src
        // If we want T_relative to be T_cur_ref (Ref to Cur), then T_cur_ref = T_cur^{-1} ? No.
        // T_cur_ref means P_cur = T_cur_ref * P_ref.
        // Conventional VINS T_w_i. P_w = T_w_i * P_i.
        // P_cur = T_w_cur^{-1} * P_w = T_w_cur^{-1} * T_w_ref * P_ref.
        // So T_cur_ref = T_w_cur^{-1} * T_w_ref.
        
        // In PhotometricRegFactor:
        // T_relative = R_tgt^T * R_src ... this is indeed T_cur_ref if pose_tgt=T_w_cur and pose_src=T_w_ref.
        
        // So:
        // pose_src (Ref) = Identity (Origin)
        // pose_tgt (Cur) = T_ref_cur (Cur w.r.t Ref) ? No.
        // If pose_src = I, then T_w_src = I.
        // T_relative = T_w_tgt^{-1} * I = T_w_tgt^{-1}.
        // This is T_tgt_w (from World(Ref) to Target).
        // This is exactly T_cur_ref (Ref is World).
        
        // So if we set pose_src=Identity and optimize pose_tgt, the resulting pose_tgt is T_ref_cur (pose of Cur in Ref frame).
        // And T_relative computed inside factor will be T_cur_ref.
        
        // WAIT! Standard VINS pose is T_w_b (Body to World).
        // pose_tgt IS T_ref_cur (Position of Cur in Ref frame).
        // Then T_w_tgt (if world is Ref) is T_ref_cur.
        // Factor computes T_relative = T_w_tgt^{-1} * T_w_src
        // = T_ref_cur^{-1} * I = T_cur_ref.
        // This checks out.
        
        // So:
        // Parameter Block 0 (Ref): Fixed at Identity.
        // Parameter Block 1 (Cur): Optimized, initialized with T_ref_cur? NO.
        // The factor takes global poses.
        // If we want to optimize relative pose T_cur_ref directly?
        // We act AS IF 'Ref' is at global origin.
        // Then 'Cur' is at T_ref_cur (inverse of T_cur_ref).
        // Let's optimize T_ref_cur (Pose of Cur in Ref).
        
        // Initial guess for T_ref_cur:
        // task.q_initial / t_initial are T_cur_ref (Ref -> Cur transform).
        // Sparse Factor expects T_cur_ref (Point Ref -> Point Cur).
        // Dense Factor expects T_ref_cur (Pose of Cur in Ref) and internally inverts it.
        
        Eigen::Quaterniond q_cur_ref = task.q_initial;
        Eigen::Vector3d t_cur_ref = task.t_initial;
        
        // Default to T_ref_cur (Dense Mode Logic)
        Eigen::Quaterniond q_init_opt = q_cur_ref.inverse();
        Eigen::Vector3d t_init_opt = -(q_init_opt * t_cur_ref);
        
        if (use_sparse_) {
             // Sparse Mode: Optimize T_cur_ref directly
             q_init_opt = q_cur_ref;
             t_init_opt = t_cur_ref;
        }

        double p_ref[7] = {0,0,0, 0,0,0,1}; // Identity
        // Initialize p_cur with the correct transform for the mode
        Eigen::Vector3d t_pert = t_init_opt;
        Eigen::Quaterniond q_pert = q_init_opt;
        
       

        double p_cur[7] = {t_pert.x(), t_pert.y(), t_pert.z(), 
                           q_pert.x(), q_pert.y(), q_pert.z(), q_pert.w()};
                           
        // GN tracking variables (replaces ceres::Solver::Summary)
        int num_factors = 0;
        double gn_initial_cost = 0, gn_final_cost = 0;
        int gn_iterations = 0;
        Eigen::Matrix<double, 6, 6> gn_hessian = Eigen::Matrix<double, 6, 6>::Zero();
        double affine[2] = {1.0, 0.0};
        double scale_shift[2] = {task.init_scale, task.init_shift};

        ceres::Solver::Summary summary; // Only used for dense mode fallback

        if (use_sparse_) {
            // --- COARSE-TO-FINE PYRAMID (DSO-style) ---
            const int NUM_LEVELS = 1;
            printf("[Pyramid] Building %d-level pyramid. img_ref: %dx%d, depth: %dx%d (type=%d, empty=%d)\n",
                   NUM_LEVELS, task.img_ref.cols, task.img_ref.rows,
                   task.depth_ref.cols, task.depth_ref.rows, task.depth_ref.type(), task.depth_ref.empty());

            // Build Image Pyramids
            std::vector<cv::Mat> pyr_ref(NUM_LEVELS), pyr_cur(NUM_LEVELS), pyr_depth(NUM_LEVELS);
            std::vector<Eigen::Matrix3d> pyr_K(NUM_LEVELS);

            pyr_ref[0] = task.img_ref;
            pyr_cur[0] = task.img_cur;
            pyr_depth[0] = task.depth_ref;
            pyr_K[0] = task.K;

            for (int l = 1; l < NUM_LEVELS; l++) {
                cv::pyrDown(pyr_ref[l-1], pyr_ref[l]);
                cv::pyrDown(pyr_cur[l-1], pyr_cur[l]);
                cv::Mat depth_down;
                cv::resize(pyr_depth[l-1], depth_down,
                          cv::Size(pyr_depth[l-1].cols/2, pyr_depth[l-1].rows/2),
                          0, 0, cv::INTER_NEAREST);
                pyr_depth[l] = depth_down;

                pyr_K[l] = pyr_K[l-1];
                pyr_K[l](0,0) *= 0.5;
                pyr_K[l](1,1) *= 0.5;
                pyr_K[l](0,2) *= 0.5;
                pyr_K[l](1,2) *= 0.5;
                printf("[Pyramid] Level %d: %dx%d, depth %dx%d\n", l, pyr_ref[l].cols, pyr_ref[l].rows, pyr_depth[l].cols, pyr_depth[l].rows);
            }

            // Iterate from COARSEST to FINEST
            for (int level = NUM_LEVELS - 1; level >= 0; level--) {
                printf("[Pyramid] Starting level %d\n", level);
                const cv::Mat& img_ref_l = pyr_ref[level];
                const cv::Mat& img_cur_l = pyr_cur[level];
                const cv::Mat& depth_l = pyr_depth[level];
                const Eigen::Matrix3d& K_l = pyr_K[level];

                // 1. Select Features (with pattern expansion)
                std::vector<FeaturePoint> features;
                int grid = (level == 0) ? 24 : 14;
                int per_cell = (level == 0) ? 2 : 1;
                selectPixelFeatures(img_ref_l, depth_l, features, grid, per_cell);

                if (features.size() < 10) {
                    printf("[Refinement] Level %d: too few features (%zu), skipping\n", level, features.size());
                    continue;
                }

                Eigen::Quaterniond q_level(p_cur[6], p_cur[3], p_cur[4], p_cur[5]);
                Eigen::Vector3d t_level(p_cur[0], p_cur[1], p_cur[2]);
                bool do_affine = (mode_ == POSE_AFFINE || mode_ == POSE_FULL || mode_ == POSE_FULL_SHIFT);
                bool do_ss     = (mode_ == POSE_SCALE_SHIFT || mode_ == POSE_FULL || mode_ == POSE_FULL_SHIFT);
                bool do_shift  = (mode_ == POSE_FULL_SHIFT);

                double init_aff[2] = { affine[0], affine[1] };
                double init_ss[2]  = { scale_shift[0], scale_shift[1] };
                auto gn = solvePhotometricGN_Extended(
                    features, img_cur_l, K_l,
                    q_level, t_level,
                    50,
                    do_affine,
                    do_ss,
                    init_aff,
                    init_ss,
                    do_shift
                );

                p_cur[0] = gn.t_cur_ref.x();
                p_cur[1] = gn.t_cur_ref.y();
                p_cur[2] = gn.t_cur_ref.z();
                p_cur[3] = gn.q_cur_ref.x();
                p_cur[4] = gn.q_cur_ref.y();
                p_cur[5] = gn.q_cur_ref.z();
                p_cur[6] = gn.q_cur_ref.w();
                if (do_affine) {
                    affine[0] = gn.alpha;
                    affine[1] = gn.beta;
                }
                if (do_ss) {
                    scale_shift[0] = gn.scale;
                    scale_shift[1] = gn.shift;
                }

                num_factors = gn.num_valid;
                gn_initial_cost = gn.initial_cost;
                gn_final_cost = gn.final_cost;
                gn_iterations = gn.iterations;
                gn_hessian = gn.H_full.topLeftCorner<6,6>();
            }

            // --- Dense Depth-to-Depth Scale/Shift Refinement ---
            if (!task.depth_cur.empty() &&
                (mode_ == POSE_SCALE_SHIFT || mode_ == POSE_FULL || mode_ == POSE_FULL_SHIFT)) {
                Eigen::Quaterniond q_refined(p_cur[6], p_cur[3], p_cur[4], p_cur[5]);
                Eigen::Matrix3d R_refined = q_refined.toRotationMatrix();
                Eigen::Vector3d t_refined(p_cur[0], p_cur[1], p_cur[2]);

                auto ss_result = refineDenseScaleShift(
                    task.depth_ref, task.depth_cur,
                    R_refined, t_refined, task.K,
                    scale_shift[0], scale_shift[1],
                    task.init_scale_cur, task.init_shift_cur);

                if (ss_result.num_valid >= 100) {
                    printf("[DenseSS] Refined: s=%.4f->%.4f, h=%.6f->%.6f\n",
                           scale_shift[0], ss_result.scale, scale_shift[1], ss_result.shift);
                    scale_shift[0] = ss_result.scale;
                    scale_shift[1] = ss_result.shift;
                }
            }

        } else {
            // --- DENSE MODE (Legacy, single level) ---
            ceres::Problem problem;
            ceres::LossFunction* loss_function = new ceres::HuberLoss(huber_loss_);

            switch(mode_) {
                case POSE_ONLY: {
                    ceres::CostFunction* cost_function = PhotometricRegFactor::Create(
                        task.img_ref, task.img_cur, task.depth_ref, task.K,
                        1.0, ssim_weight_, l1_weight_);
                    problem.AddResidualBlock(cost_function, loss_function, p_ref, p_cur);
                    break;
                }

                case POSE_AFFINE: {
                    ceres::CostFunction* cost_function_affine = PhotometricRegFactorAffine::Create(
                        task.img_ref, task.img_cur, task.depth_ref, task.K,
                        1.0, ssim_weight_, l1_weight_);
                    problem.AddResidualBlock(cost_function_affine, loss_function, p_ref, p_cur, affine);
                    break;
                }

                case POSE_SCALE_SHIFT: {
                    ceres::CostFunction* cost_function_ss = PhotometricRegFactorScaleShift::Create(
                        task.img_ref, task.img_cur, task.depth_ref, task.K,
                        1.0, ssim_weight_, l1_weight_);
                    problem.AddResidualBlock(cost_function_ss, loss_function, p_ref, p_cur, scale_shift);
                    break;
                }

                default: {
                    ceres::CostFunction* cost_function = PhotometricRegFactor::Create(
                        task.img_ref, task.img_cur, task.depth_ref, task.K,
                        1.0, ssim_weight_, l1_weight_);
                    problem.AddResidualBlock(cost_function, loss_function, p_ref, p_cur);
                    break;
                }
            }

            if (problem.HasParameterBlock(p_ref)) {
                 problem.SetParameterBlockConstant(p_ref);
            }
            if (mode_ == POSE_SCALE_SHIFT) {
                 if (problem.HasParameterBlock(p_cur)) {
                    problem.SetParameterBlockConstant(p_cur);
                 }
            }

            auto* manifold = new ceres::ProductManifold<
                ceres::EuclideanManifold<3>,
                ceres::EigenQuaternionManifold>();
            problem.SetManifold(p_cur, manifold);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 30;
            options.minimizer_progress_to_stdout = false;
            options.num_threads = 1;

            ceres::Solve(options, &problem, &summary);
        }

        // Extract result
        RefinementResult result;
        result.timestamp_ref = task.t_ref;
        result.timestamp_cur = task.t_cur;

        // Unify cost variables: use GN for sparse, ceres summary for dense
        double opt_initial_cost = use_sparse_ ? gn_initial_cost : summary.initial_cost;
        double opt_final_cost = use_sparse_ ? gn_final_cost : summary.final_cost;
        int opt_iterations = use_sparse_ ? gn_iterations : (int)summary.iterations.size();
        bool opt_success = use_sparse_ ? (gn_final_cost < gn_initial_cost || gn_iterations > 0)
                                       : (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < summary.initial_cost);
        
        if (opt_success) {
            // Extract result
            Eigen::Vector3d t_res(p_cur[0], p_cur[1], p_cur[2]);
            Eigen::Quaterniond q_res(p_cur[6], p_cur[3], p_cur[4], p_cur[5]);
            if (use_sparse_) {
                 Eigen::Quaterniond q_inv = q_res.inverse();
                 Eigen::Vector3d t_inv = -(q_inv * t_res);
                 result.t_ref_cur = t_inv;
                 result.q_ref_cur = q_inv;
            } else {
                 result.t_ref_cur = t_res;
                 result.q_ref_cur = q_res;
            }

            // Standardize Quaternion Sign (w > 0)
            if (result.q_ref_cur.w() < 0) {
                result.q_ref_cur.coeffs() = -result.q_ref_cur.coeffs();
            }

            result.alpha = affine[0];
            result.beta = affine[1];
            result.scale = scale_shift[0];
            result.shift = scale_shift[1];
            result.feature_count = num_factors;
            result.initial_cost = gn_initial_cost;
            result.final_cost = gn_final_cost;
            result.success = true;

            // Calculate Hessian / Information Matrix
            if (use_sparse_) {
                // Single global scale preserving eigenstructure
                double trace_full = gn_hessian.trace();
                double global_scale = (trace_full > 0) ? 6000.0 / trace_full : 1000.0;
                result.information = gn_hessian * global_scale;
            } else {
                result.information = Eigen::Matrix<double, 6, 6>::Identity() * 2.0e4;
            }

            Eigen::Vector3d t_vins = task.t_initial;
            Eigen::Quaterniond q_vins = task.q_initial;
            Eigen::Quaterniond q_opt_cur_ref;
            Eigen::Vector3d t_opt_cur_ref;
            
            if (use_sparse_) {
                 t_opt_cur_ref = t_res;
                 q_opt_cur_ref = q_res;
            } else {
                 q_opt_cur_ref = q_res.inverse();
                 t_opt_cur_ref = -(q_opt_cur_ref * t_res);
            }
            
            auto t_end = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            
            printf("\n[Photometric Refinement] Mode: %d | Frame: %.3f -> %.3f | Time: %.2f ms\n", mode_, task.t_ref, task.t_cur, duration_ms);
            printf("  Iterations: %d | Cost: %.4e -> %.4e\n", opt_iterations, opt_initial_cost, opt_final_cost);
            printf("  VINS Pose (T_cur_ref): t=[%.3f, %.3f, %.3f] q=[%.3f, %.3f, %.3f, %.3f]\n",
                   t_vins.x(), t_vins.y(), t_vins.z(), q_vins.w(), q_vins.x(), q_vins.y(), q_vins.z());
            printf("  Ours Pose (T_cur_ref): t=[%.3f, %.3f, %.3f] q=[%.3f, %.3f, %.3f, %.3f]\n",
                   t_opt_cur_ref.x(), t_opt_cur_ref.y(), t_opt_cur_ref.z(), 
                   q_opt_cur_ref.w(), q_opt_cur_ref.x(), q_opt_cur_ref.y(), q_opt_cur_ref.z());
            // --- ADD THIS BLOCK ---
            if (mode_ == POSE_AFFINE || mode_ == POSE_FULL || mode_ == POSE_FULL_SHIFT) {
                printf("  Affine: alpha=%.5f, beta=%.5f\n", result.alpha, result.beta);
            }
            if (mode_ == POSE_SCALE_SHIFT || mode_ == POSE_FULL || mode_ == POSE_FULL_SHIFT) {
                printf("  Scale/Shift: scale=%.5f, shift=%.5f\n", result.scale, result.shift);
            }
            // ----------------------
            Eigen::Vector3d dt_diag = t_opt_cur_ref - t_vins;
            Eigen::Quaterniond dq_diag = q_opt_cur_ref * q_vins.inverse();
            Eigen::AngleAxisd aa(dq_diag);
            double angle_deg = aa.angle() * 180.0 / M_PI;
            printf("  DELTA: dt=[%.6f, %.6f, %.6f] mm  |dt|=%.4f mm  angle=%.4f deg\n",
                   dt_diag.x()*1000, dt_diag.y()*1000, dt_diag.z()*1000, dt_diag.norm()*1000, angle_deg);
            printf("  Cost reduction: %.2f%%\n", 
                   (opt_initial_cost > 0) ? 100.0 * (1.0 - opt_final_cost / opt_initial_cost) : 0.0);
            
        } else {
            result.success = false;
            printf("[Photometric Refinement] FAILED. Cost: %.4e -> %.4e\n", opt_initial_cost, opt_final_cost);
        }

        // --- KAIZEN: Guard Rails / Gating ---
        if (result.success) {
            double dt = std::abs(result.timestamp_cur - result.timestamp_ref);
            if (dt < 1e-4) dt = 1e-4; // Avoid div by zero
            
            double v_est = result.t_ref_cur.norm() / dt;
            
            Eigen::AngleAxisd aa(result.q_ref_cur);
            double rot_angle = aa.angle();
            
            // Thresholds: 5 m/s, 0.5 rad (~30 deg)
            bool velocity_check = (v_est < 5.0); 
            bool rotation_check = (std::abs(rot_angle) < 0.5);
            bool cost_check = (opt_final_cost < opt_initial_cost);
            
            if (!velocity_check) {
                result.success = false;
                printf("\033[1;31m[Photometric Refinement] REJECTED (Velocity): v=%.2f m/s (> 5.0)\033[0m\n", v_est);
            } else if (!rotation_check) {
                result.success = false;
                printf("\033[1;31m[Photometric Refinement] REJECTED (Rotation): angle=%.2f rad (> 0.5)\033[0m\n", rot_angle);
            } else if (!cost_check) {
                 result.success = false;
                 printf("\033[1;31m[Photometric Refinement] REJECTED (Cost): Cost increased or flat\033[0m\n");
            } else {
                 printf("\033[1;32m[Photometric Refinement] ACCEPTED. v=%.2f m/s, angle=%.2f rad\033[0m\n", v_est, rot_angle);
            }
        }
        
        // Push result
        {
            std::lock_guard<std::mutex> lock(result_mutex_1_);
            result_queue_1_.push(result);
        }
        
        // Clear busy flag
        thread_1_busy_.store(false);
    }
}
}
