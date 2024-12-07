
__kernel void ImageCopy(read_only image2d_t image,__global uchar* output_image, long width, long height) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;    
    const int target_index = 2048;

    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    uint4 p = read_imageui(image, sampler,(int2)(pos.x, pos.y));
    int index = get_global_id(0) +  get_global_id(1) * width;

    uchar pattern[32];
    uchar min_value[16];
    uchar max_value[16];

    int width_offset[16] = {
        0, 1, 2, 3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1
    };

    int height_offset[16] = {
        -3, -3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3
    };

    for (int i = 0; i < 16; i++) {
        uint4 pixel = read_imageui(image, sampler,(int2)(pos.x + width_offset[i], pos.y + height_offset[i]));
        pattern[i] = pattern[i + 16] = pixel.x;
        if (index == target_index) {
            printf("Pixel[%d] : %d\n", i, pixel.x);
        }
    }

    int score = 0;
    for(int i = 0; i < 16; i++) {
        min_value[i] = pattern[i];
        max_value[i] = pattern[i];
        for(int j = 1; j < 9; j++) {
            min_value[i] = min(min_value[i], pattern[i + j]);
            max_value[i] = max(max_value[i], pattern[i + j]);
        }
    }

    for (int i = 0; i < 16; i++) {
        min_value[0] = max(min_value[0], min_value[i]);
        max_value[0] = min(max_value[0], max_value[i]);
    }

    if (index == target_index) {
        printf("min %d, max %d ,p value %d \n",min_value[0], max_value[0], p.x);
        printf("min - p = %d, p - max = %d ,p value %d \n",min_value[0] - p.x, p.x - max_value[0], p.x);

    }

    score = max(min_value[0] - p.x, p.x - max_value[0]);
    if (target_index == index) {
        printf("Score %d\n", score);
    }

    if (score > 10) {
        output_image[index] = 255;
        printf("Greater\n");
    } else {
        output_image[index] = 0;
    }
    output_image[index] = p.x;
}

__kernel void FASTCorner(read_only image2d_t image, __global uchar* output_image, int threshold) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int index = pos.y * get_global_size(0) + pos.x;
    
    // Skip border pixels
    if(pos.x < 3 || pos.y < 3 || pos.x >= get_image_width(image)-3 || pos.y >= get_image_height(image)-3) {
        output_image[index] = 0;
        return;
    }

    uint4 center = read_imageui(image, sampler, pos);
    uchar center_val = center.x;
    
    // Circle pixels around center point
    uchar circle[16];
    
    // Offsets for circle pixels
    int width_offset[16] = {
        0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1
    };
    
    int height_offset[16] = {
        -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3
    };

    // Read circle pixels
    for(int i = 0; i < 16; i++) {
        uint4 pixel = read_imageui(image, sampler, (int2)(pos.x + width_offset[i], pos.y + height_offset[i]));
        circle[i] = pixel.x;
    }

    // Calculate corner score
    int max_score = 0;
    
    // Check each arc of 9 contiguous pixels
    for(int i = 0; i < 16; i++) {
        int min_val = 255;
        int max_val = 0;
        
        // Check 9 contiguous pixels
        for(int j = 0; j < 9; j++) {
            int idx = (i + j) % 16;
            min_val = min(min_val, (int)circle[idx]);
            max_val = max(max_val, (int)circle[idx]);
        }
        
        // Update max score
        max_score = max(max_score, 
                       max(min_val - (int)center_val, (int)center_val - max_val));
    }
    
    // Only mark as corner if score exceeds threshold
    output_image[index] = (max_score > threshold) ? max_score : 0;
}

__kernel void NonMaximumSuppression(__global uchar* image, __global uchar* output_image, int radius) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int width = get_global_size(0);
    int height = get_global_size(1);
    int index = pos.y * width + pos.x;
    
    // Skip border pixels
    if(pos.x < radius || pos.y < radius || 
       pos.x >= width-radius || 
       pos.y >= height-radius) {
        output_image[index] = 0;
        image[index] = 0;
        return;
    }

    int center_val = image[index];

    // If center pixel is not a corner, skip
    if(center_val == 0) {
        output_image[index] = 0;
        image[index] = 0;
        return;
    }

    // Check if center is local maximum in radius neighborhood
    for(int dy = -radius; dy <= radius; dy++) {
        for(int dx = -radius; dx <= radius; dx++) {
            // Skip center pixel
            if(dx == 0 && dy == 0) continue;
            
            int neighbor_idx = (pos.y + dy) * width + (pos.x + dx);
            int neighbor_val = image[neighbor_idx];
            
            if(neighbor_val > center_val) {
                output_image[index] = 0;
                image[index] = 0;
                return;
            }
        }
    }

    // If we get here, center is local maximum
    output_image[index] = center_val;
}   

