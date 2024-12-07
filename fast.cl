
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

    // Check for contiguous brighter pixels
    int bright_count = 0;
    int max_bright = 0;
    
    for(int i = 0; i < 32; i++) {
        int idx = i % 16;
        if(circle[idx] >= center_val + threshold) {
            bright_count++;
            max_bright = max(max_bright, bright_count);
        } else {
            bright_count = 0;
        }
    }
    
    if(max_bright >= 9) {
        output_image[index] = 255;
        return;
    }
    
    // Check for contiguous darker pixels
    int dark_count = 0;
    int max_dark = 0;
    
    for(int i = 0; i < 32; i++) {
        int idx = i % 16;
        if(circle[idx] <= center_val - threshold) {
            dark_count++;
            max_dark = max(max_dark, dark_count);
        } else {
            dark_count = 0;
        }
    }
    
    output_image[index] = (max_dark >= 9) ? 255 : 0;
}

