import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# --- CONFIGURACIÓN ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "models", "nandi_v4_final_history.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "v4_training_curves.png")


def plot_training_curves():
    if not os.path.exists(CSV_PATH):
        print(f"❌ No se encuentra {CSV_PATH}")
        print("   Asegurate de haber corrido train_model.py primero.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    epochs = range(1, len(df) + 1)

    # --- ESTILO OSCURO ---
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0e1117")

    COLOR_TRAIN = "#4FC3F7"   # azul claro
    COLOR_VAL   = "#FF7043"   # naranja
    COLOR_GRID  = "#2a2a2a"

    # --- GRÁFICO 1: Accuracy ---
    ax1.set_facecolor("#0e1117")
    ax1.plot(epochs, df["accuracy"],     color=COLOR_TRAIN, linewidth=2,   label="Train")
    ax1.plot(epochs, df["val_accuracy"], color=COLOR_VAL,   linewidth=2,   label="Validación", linestyle="--")

    # Marcar el mejor val_accuracy
    best_epoch = df["val_accuracy"].idxmax() + 1
    best_val   = df["val_accuracy"].max()
    ax1.scatter(best_epoch, best_val, color=COLOR_VAL, s=100, zorder=5)
    ax1.annotate(
        f"  Best: {best_val*100:.1f}% (época {best_epoch})",
        xy=(best_epoch, best_val),
        color=COLOR_VAL,
        fontsize=9
    )

    ax1.set_title("Accuracy", color="white", fontsize=13, pad=12)
    ax1.set_xlabel("Época", color="white")
    ax1.set_ylabel("Accuracy", color="white")
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax1.tick_params(colors="white")
    ax1.grid(color=COLOR_GRID, linestyle="--", linewidth=0.5)
    ax1.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")
    ax1.set_ylim(0, 1.05)

    # --- GRÁFICO 2: Loss ---
    ax2.set_facecolor("#0e1117")
    ax2.plot(epochs, df["loss"],     color=COLOR_TRAIN, linewidth=2, label="Train")
    ax2.plot(epochs, df["val_loss"], color=COLOR_VAL,   linewidth=2, label="Validación", linestyle="--")

    # Marcar el mejor val_loss
    best_loss_epoch = df["val_loss"].idxmin() + 1
    best_loss       = df["val_loss"].min()
    ax2.scatter(best_loss_epoch, best_loss, color=COLOR_VAL, s=100, zorder=5)
    ax2.annotate(
        f"  Best: {best_loss:.3f} (época {best_loss_epoch})",
        xy=(best_loss_epoch, best_loss),
        color=COLOR_VAL,
        fontsize=9
    )

    ax2.set_title("Loss", color="white", fontsize=13, pad=12)
    ax2.set_xlabel("Época", color="white")
    ax2.set_ylabel("Loss", color="white")
    ax2.tick_params(colors="white")
    ax2.grid(color=COLOR_GRID, linestyle="--", linewidth=0.5)
    ax2.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")

    # --- TÍTULO GENERAL ---
    fig.suptitle(
        "Nandi AI V4 — Curvas de Entrenamiento  |  Best val_accuracy: 97.33%",
        color="white", fontsize=14, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"✅ Curva guardada en: {OUTPUT_PATH}")
    print(f"   Mejor val_accuracy: {best_val*100:.2f}% en época {best_epoch}")


if __name__ == "__main__":
    plot_training_curves()