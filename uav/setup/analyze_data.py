def analyze_data():
    print("to be implemented.")



# amount of images, amount of annotations, per modality! (vz, ir)
# some ratios, balance ratio?

# { datasets/anti-uav300/statistics.json each is sequence: [numvisible images, num infrared images, num visible annotations, num infrared annotations]
    # "20190925_200805_1_2": [
    #     934,
    #     934,
    #     934,
    #     934
    # ],
    # "20190925_200805_1_5": [
    #     1000,
    #     1000,
    #     1000,
    #     944
    # ],
    # "20190925_111757_1_4": [
    #     1000,
    #     1000,
    #     1000,
    #     1000
    # ],
    # "20190925_111757_1_3": [
    #     1000,
    #     1000,
    #     1000,
    #     1000
    # ],


# save somewhere. allow plotting! use visualization in script instead of in seperate dir. code below!

# import matplotlib.pyplot as plt

# frames_vz = 296901
# frames_ir = 296901
# annot_vz = 280218
# annot_ir = 293209

# fig, ax = plt.subplots(figsize=(10, 5))
# colors = ['#2E86AB', '#A23B72']

# ratio_vz = annot_vz / frames_vz
# ratio_ir = annot_ir / frames_ir

# bar_width = 1.0
# x_positions = [0, 1]

# # Background bars (total frames)
# ax.bar(x_positions, [frames_vz, frames_ir], 
#        width=bar_width, color=colors, alpha=0.2, 
#        label='Total Frames', edgecolor='grey', linewidth=0.5)

# # Foreground bars (annotated frames)
# ax.bar(x_positions, [annot_vz, annot_ir], 
#        width=bar_width, color=colors, alpha=0.8,
#        label='Annotated Frames', edgecolor='black', linewidth=0.5)

# # Labels
# for i, (fv, av, rv) in enumerate(zip([frames_vz, frames_ir], 
#                                     [annot_vz, annot_ir], 
#                                     [ratio_vz, ratio_ir])):
#     ax.text(i, fv * 0.5, f'({rv:.2%})', 
#             ha='center', va='center', fontweight='bold', 
#             fontsize=11, color='white')

# # Styling
# ax.set_xlabel('Modality', fontsize=12, labelpad=10)
# ax.set_ylabel('Number of Frames', fontsize=12, labelpad=10)
# ax.set_title('Annotation Coverage by Modality', fontsize=14, pad=20)

# ax.set_xticks(x_positions)
# ax.set_xticklabels(['Visible Spectrum', 'Infrared'])

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.grid(axis='y', alpha=0.3, linestyle='--')

# plt.tight_layout()
# plt.savefig("visualization/dataset_modality_coverage.png", dpi=300)
