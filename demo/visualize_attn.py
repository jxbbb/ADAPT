import numpy as np
from visualize import visualize_grid_attention_v2
import os
import shutil

attention_probs = np.load("attn_visualize/head_135_11/atten.npy")
word_num = len(attention_probs)-784

# attention_probs = np.expand_dims(attention_probs, axis=0)

last_word_attention = attention_probs[17][word_num:].reshape(16, 7, 7)

for t_i in range(16):
    img_attn = last_word_attention[t_i]

    save_path = f"attn_visualize_one_video/{t_i}"
    os.makedirs(save_path, exist_ok=True)

    frameid = 2*t_i
    shutil.copyfile(f"demo/{frameid}.png", save_path+f"/{frameid}.png")
    visualize_grid_attention_v2(f"demo/{frameid}.png",
                                    save_path=save_path,
                                    attention_mask=img_attn,
                                    save_image=True,
                                    save_original_image=False,
                                    quality=100)

print(f"word_num: {word_num}")