
import os

target_file = "/media/SSD/photo/vinsfusion-clean/src/vins_estimator/estimator/estimator.cpp"

with open(target_file, 'r') as f:
    lines = f.readlines()

start_marker = "// [Refinement] Submit Task if Keyframe and Initialized"
end_marker = "photometricRefinement->submitTask"

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if start_marker in line:
        start_idx = i
    if end_marker in line and start_idx != -1:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    cnt = 0
    cutoff_idx = end_idx + 1
    while cnt < 3 and cutoff_idx < len(lines):
        if "}" in lines[cutoff_idx]:
            cnt += 1
        cutoff_idx += 1
    
    # New content
    new_block = []
    new_block.append("  // [Refinement] Submit Task if Keyframe and Initialized - Optimize 7 and 8\n")
    new_block.append("  if (solver_flag == NON_LINEAR && photometricRefinement && marginalization_flag == MARGIN_OLD) {\n")
    new_block.append("      int idx_ref = 7;\n")
    new_block.append("      int idx_cur = 8;\n")
    new_block.append("      \n")
    new_block.append("      // Ensure we have enough frames\n")
    new_block.append("      if (frame_count >= WINDOW_SIZE) { \n")
    new_block.append("          double t_ref = Headers[idx_ref];\n")
    new_block.append("          double t_cur = Headers[idx_cur];\n")
    new_block.append("          \n")
    new_block.append("          cv::Mat img_ref, img_cur, depth_ref;\n")
    new_block.append("          \n")
    new_block.append("          if (buffer_images_.count(t_ref) && buffer_images_.count(t_cur) && buffer_depths_.count(t_ref)) {\n")
    new_block.append("               img_ref = buffer_images_[t_ref];\n")
    new_block.append("               img_cur = buffer_images_[t_cur];\n")
    new_block.append("               depth_ref = buffer_depths_[t_ref];\n")
    new_block.append("               \n")
    new_block.append("               if (!img_ref.empty() && !img_cur.empty() && !depth_ref.empty()) {\n")
    new_block.append("                    // Calculate Initial Guess based on Ps[7] and Ps[8]\n")
    new_block.append("                    Eigen::Vector3d P_ref = Ps[idx_ref];\n")
    new_block.append("                    Eigen::Matrix3d R_ref = Rs[idx_ref];\n")
    new_block.append("                    \n")
    new_block.append("                    Eigen::Vector3d P_cur = Ps[idx_cur];\n")
    new_block.append("                    Eigen::Matrix3d R_cur = Rs[idx_cur];\n")
    new_block.append("                    \n")
    new_block.append("                    Eigen::Quaterniond Q_ref(R_ref);\n")
    new_block.append("                    Eigen::Quaterniond Q_cur(R_cur);\n")
    new_block.append("                    \n")
    new_block.append("                    // DEBUG: Print World Poses\n")
    new_block.append("                    printf(\"[Estimator] PhotoRef Init (7->8): P_7(Ref)=%.3f,%.3f,%.3f  P_8(Cur)=%.3f,%.3f,%.3f\\n\", \n")
    new_block.append("                            P_ref.x(), P_ref.y(), P_ref.z(), P_cur.x(), P_cur.y(), P_cur.z());\n")
    new_block.append("            \n")
    new_block.append("                    // IMU Relative Pose: T_cur_ref (Ref in Cur frame)\n")
    new_block.append("                    Eigen::Quaterniond Q_cur_inv = Q_cur.inverse();\n")
    new_block.append("                    Eigen::Quaterniond q_b_cur_ref = Q_cur_inv * Q_ref;\n")
    new_block.append("                    Eigen::Vector3d t_b_cur_ref = Q_cur_inv * (P_ref - P_cur);\n")
    new_block.append("                    \n")
    new_block.append("                    // Convert to Camera Frame (Cam 0)\n")
    new_block.append("                    // T_c_cur_ref = T_c_b * T_b_cur_ref * T_b_c\n")
    new_block.append("                    // ric, tic are T_b_c (Body->Cam)\n")
    new_block.append("                    Eigen::Matrix3d R_b_c = ric[0];\n")
    new_block.append("                    Eigen::Vector3d t_b_c = tic[0];\n")
    new_block.append("                    \n")
    new_block.append("                    Eigen::Matrix3d R_c_b = R_b_c.transpose();\n")
    new_block.append("                    Eigen::Vector3d t_c_b = -R_c_b * t_b_c;\n")
    new_block.append("                    \n")
    new_block.append("                    Eigen::Quaterniond Q_c_b(R_c_b);\n")
    new_block.append("                    Eigen::Quaterniond Q_b_c(R_b_c);\n")
    new_block.append("                    \n")
    new_block.append("                    // T_c_cur_ref = T_c_b * T_b_cur_ref * T_b_c\n")
    new_block.append("                    Eigen::Quaterniond q_initial = Q_c_b * q_b_cur_ref * Q_b_c;\n")
    new_block.append("                    Eigen::Vector3d t_initial = Q_c_b * (q_b_cur_ref * t_b_c + t_b_cur_ref) + t_c_b;\n")
    new_block.append("                   \n")
    new_block.append("                   // Global params for K\n")
    new_block.append("                   Eigen::Matrix3d K;\n")
    new_block.append("                   K << params.fx, 0, params.cx,\n")
    new_block.append("                        0, params.fy, params.cy,\n")
    new_block.append("                        0, 0, 1;\n")
    new_block.append("                        \n")
    new_block.append("                   photometricRefinement->submitTask(t_ref, t_cur, img_ref, img_cur, depth_ref, K, q_initial, t_initial);\n")
    new_block.append("               }\n")
    new_block.append("          }\n")
    new_block.append("      }\n")
    new_block.append("  }\n")

    final_lines = lines[:start_idx] + new_block + lines[cutoff_idx:]
    
    with open(target_file, 'w') as f:
        f.writelines(final_lines)
    print("Successfully patched.")
else:
    print("Could not find markers.")
