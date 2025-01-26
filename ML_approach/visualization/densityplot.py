import matplotlib.pyplot as plt
import scipy.stats as kde 


# MAE Beginn
plt.figure(figsize=(8, 6))

#point size und transparency
plt.scatter(y_test, y_pred, s=10, alpha=0.3, color='green', label='Predictions')

# density contour
xy = np.vstack([y_test, y_pred])
z = kde.gaussian_kde(xy)(xy)
plt.scatter(y_test, y_pred, c=z, s=10, cmap='viridis', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')

#labels
plt.xlabel('True Denormalized Values')
plt.ylabel('Predicted Denormalized Values')
plt.title('True vs Predicted Denormalized Values for Mean Absolute Error')

#MAE
plt.text(
    0.05, 0.95, f'MAE (Test): {test_MAE:.4f}', transform=plt.gca().transAxes,
    fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue')
)

#box ausserhalb
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#MEDAE beginn
plt.figure(figsize=(8, 6))

#point size und transparency
plt.scatter(y_test, y_pred, s=10, alpha=0.3, color='grey', label='Predictions')


z = kde.gaussian_kde(xy)(xy)
plt.scatter(y_test, y_pred, c=z, s=10, cmap='magma', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')

plt.xlabel('True Denormalized Values')
plt.ylabel('Predicted Denormalized Values')
plt.title('True vs Predicted Denormalized Values for Median Absolute Error')

# MEDAE
plt.text(
    0.05, 0.95, f'MedAE (Test): {test_MedAE:.4f}', transform=plt.gca().transAxes,
    fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue')
)

plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()