import os
import matplotlib.pyplot as plt
import tensorflow as tf

def visualize_predictions(model, dataset, num_samples=3, save_dir=None):
    """
    Visualize predictions from the trained model.

    Parameters:
    - model: Trained model to generate predictions.
    - dataset: Validation dataset (image, mask pairs).
    - num_samples: Number of batches to visualize.
    - save_dir: Directory to save visualizations (optional).
    """
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    for batch_idx, batch in enumerate(dataset.take(num_samples)):
        images, masks = batch
        preds = model.predict(images)
        preds = tf.round(preds)  # Threshold predictions to 0 or 1

        for i in range(len(images)):
            plt.figure(figsize=(12, 4))

            # Input Image
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(images[i].numpy())
            plt.axis("off")

            # True Mask
            plt.subplot(1, 3, 2)
            plt.title("True Mask")
            plt.imshow(tf.squeeze(masks[i]).numpy(), cmap="gray")
            plt.axis("off")

            # Predicted Mask
            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(tf.squeeze(preds[i]).numpy(), cmap="gray")
            plt.axis("off")

            if save_dir:
                save_path = os.path.join(save_dir, f"batch_{batch_idx}_image_{i}.png")
                plt.savefig(save_path)
                print(f"Visualization saved to {save_path}")

            plt.show()
