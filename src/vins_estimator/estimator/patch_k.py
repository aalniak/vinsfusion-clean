
import os

target_file = "/media/SSD/photo/vinsfusion-clean/src/vins_estimator/estimator/estimator.cpp"
with open(target_file, 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
patched = False

replacement = """               // FeatureTracker reads intrinsics but Estimator doesn't store K directly?
               // We have global params for this.
               Eigen::Matrix3d K;
               K << params.fx, 0, params.cx,
                    0, params.fy, params.cy,
                    0, 0, 1;
"""

for i, line in enumerate(lines):
    if "Eigen::Matrix3d K = Eigen::Matrix3d::Identity();" in line:
        # Found target line
        if not patched:
            # Look backwards to replace comments?
            # Or just replace this line and insert new block?
            # Let's just replace this line and rewrite comments if needed.
            # But the comments before it are confusing.
            # I'll replace just the K line and add comments above it.
            new_lines.append(replacement)
            patched = True
        continue
        
    # Skip lines if we are inside the block we want to remove (if any)
    # But since I'm just replacing the Identity line, no need to skip others unless I want to remove comments.
    # The previous comments are harmless but confusing.
    # Let's keep them or remove them manually if I can identify them.
    # "Actually PhotometricRefinement needs K." etc.
    if "Actually PhotometricRefinement needs K." in line:
         continue
    if "Let's assume params.focal_length" in line:
         continue
    
    new_lines.append(line)

with open(target_file, 'w') as f:
    f.writelines(new_lines)

print("Patched K" if patched else "Failed to patch K")
