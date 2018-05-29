
np.random.seed(32)
x_s = np.linspace(-4.5, 10, 200)

f, ax = plt.subplots(2, 2, figsize=(5.5, 5.5))

T0 = stats.norm(0, 1)
T1 = stats.norm(2, 0.5)
T2 = stats.norm(6, 2)

T = T0.pdf(np.sin(x_s)) * .3 + T1.pdf(x_s) * .3 + T2.pdf(x_s) * .4
T /= T.sum()
ax[0, 0].fill_between(x_s, T, alpha=1, color='C7')
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_title('True distribution')

T_sample = np.random.choice(x_s, size=20, replace=True, p=T)
ax[0, 1].plot(T_sample, np.zeros_like(T_sample) + 0.05, '.', color='C7')
ax[0, 1].set_ylim(0, 1)
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])
ax[0, 1].set_title('Sample')


az.kdeplot(T_sample, ax=ax[1, 1], bw=6, fill_alpha=1, alpha=0)
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])
ax[1, 1].set_title('Posterior distribution')

T_ppc = np.random.choice(x_s, size=40, replace=True, p=T)
az.kdeplot(T_ppc, ax=ax[1, 0], bw=6, fill_alpha=1, alpha=0)

ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 0].set_title('Predictive distribution')
f.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0.5)

ax[1, 0].text(12.5, 0.29, "sampling")
ax[1, 0].annotate('', xy=(18, 0.27), xytext=(12, 0.27),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 annotation_clip=False)

ax[1, 0].text(28, 0.19, "inference")
ax[1, 0].annotate('', xy=(27, 0.16), xytext=(27, 0.21),
                  ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 annotation_clip=False)

ax[1, 0].text(12.5, 0.08, "prediction")
ax[1, 0].annotate('', xy=(12, 0.06), xytext=(18, 0.06),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 annotation_clip=False)

ax[1, 0].text(9, 0.18, "validation")
ax[1, 0].annotate('', xy=(18.5, 0.21), xytext=(11, 0.14),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 annotation_clip=False)

plt.savefig('B11197_01_08.png', dpi=300)
