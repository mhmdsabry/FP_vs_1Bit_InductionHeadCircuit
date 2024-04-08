import os
import matplotlib.pyplot as plt
import torch

# Directory containing all the files
dir_path = './assets/induction_scores'
average_pos_dir = 'train_avg_pos_effects'
average_norm_dir = 'train_avg_norm_effects'

full_average_pos_path = os.path.join(dir_path, average_pos_dir)
full_average_norm_path = os.path.join(dir_path, average_norm_dir)

eval_average_pos_dir = 'eval_avg_pos_effects'
eval_average_norm_dir = 'eval_avg_norm_effects'

full_eval_average_pos_path = os.path.join(dir_path, eval_average_pos_dir)
full_eval_average_norm_path = os.path.join(dir_path, eval_average_norm_dir)

pos_types = ["trigonometric", "learnable", "no_pos"]
norm_types = ["post_norm","pre_norm", "no_norm"]

data = {
    'FP': {},
    '1Bit': {},
}

def plot_avg_effect(effect_types, full_average_effect_path, mode="train"):
    for effect_type in effect_types:
        files_path = os.path.join(full_average_effect_path, effect_type)
        for file_name in os.listdir(files_path):
            file_components = file_name.split('_')
            P_type, embed_size, head_num = file_components[0], file_components[2], file_components[4]
            embed_size, head_num = int(embed_size), int(head_num)
    
            value = torch.load(os.path.join(files_path, file_name))

            if P_type not in data:
                data[P_type] = {}
            if embed_size not in data[P_type]:
                data[P_type][embed_size] = {}
            data[P_type][embed_size][head_num] = value

        embed_sizes = sorted({embed_size for type_key in data for embed_size in data[type_key]})
        head_nums = sorted({hn for type_key in data for embed_size in data[type_key] for hn in data[type_key][embed_size]})

        for P_type in data:
            fig, axs = plt.subplots(len(embed_sizes), len(head_nums), figsize=(10, 8))
            fig.suptitle(f"Avg. Induction Scores for {P_type} {effect_type}", fontsize=12)

            for i, es in enumerate(embed_sizes):
                for j, hn in enumerate(head_nums):
                    attention_scores = data[P_type][es][hn]
                    cax = axs[i][j].matshow(attention_scores.numpy(), cmap='Blues', origin='lower', aspect='auto', vmin=0, vmax=0.30)
                    axs[i][j].set_title(f'ES: {es}, HN: {hn}', fontsize=8)


            # Set common labels
            for ax in axs.flat:
                ax.set(xlabel='Head', ylabel='Layer')
            for ax in axs.flat:
                ax.label_outer()

            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.5])
            fig.colorbar(cax, cax=cbar_ax)
            plt.subplots_adjust(hspace=0.6, wspace=0.5)
            plt.savefig(f'./assets/{mode}_{P_type}_{effect_type}_average_effect.png')


#Train
#plot pos effect:
plot_avg_effect(pos_types, full_average_pos_path, mode="train")

#plot norm effect:
plot_avg_effect(norm_types, full_average_norm_path, mode="train")

#Eval
#plot pos effect:
plot_avg_effect(pos_types, full_eval_average_pos_path, mode="eval")

#plot norm effect:
plot_avg_effect(norm_types, full_eval_average_norm_path, mode="eval")