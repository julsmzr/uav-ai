import matplotlib.pyplot as plt

frames_vz = 296901
frames_ir = 296901
annot_vz = 280218
annot_ir = 293209

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#2E86AB', '#A23B72']

ratio_vz = annot_vz / frames_vz
ratio_ir = annot_ir / frames_ir

bar_width = 1.0
x_positions = [0, 1]

# Background bars (total frames)
ax.bar(x_positions, [frames_vz, frames_ir], 
       width=bar_width, color=colors, alpha=0.2, 
       label='Total Frames', edgecolor='grey', linewidth=0.5)

# Foreground bars (annotated frames)
ax.bar(x_positions, [annot_vz, annot_ir], 
       width=bar_width, color=colors, alpha=0.8,
       label='Annotated Frames', edgecolor='black', linewidth=0.5)

# Labels
for i, (fv, av, rv) in enumerate(zip([frames_vz, frames_ir], 
                                    [annot_vz, annot_ir], 
                                    [ratio_vz, ratio_ir])):
    ax.text(i, fv * 0.5, f'({rv:.2%})', 
            ha='center', va='center', fontweight='bold', 
            fontsize=11, color='white')

# Styling
ax.set_xlabel('Modality', fontsize=12, labelpad=10)
ax.set_ylabel('Number of Frames', fontsize=12, labelpad=10)
ax.set_title('Annotation Coverage by Modality', fontsize=14, pad=20)

ax.set_xticks(x_positions)
ax.set_xticklabels(['Visible Spectrum', 'Infrared'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

ax.legend(frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.show()
plt.savefig("visualization/dataset_modality_coverage.png")
