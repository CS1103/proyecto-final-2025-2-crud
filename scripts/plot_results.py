import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set(style="whitegrid")
def load_csvs(pattern):
    files = glob.glob(pattern)
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files matched the pattern: {pattern}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['__source_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
    if len(dfs) == 0:
        raise RuntimeError("No readable CSVs found")
    return pd.concat(dfs, ignore_index=True)


def plot_loss(df, outdir):
    group_cols = ['optimizer', 'lr', 'batch_size']
    df['lr'] = df['lr'].astype(str)
    df['batch_size'] = df['batch_size'].astype(str)

    agg = df.groupby(group_cols + ['epoch']).agg(
        train_loss_mean=('train_loss', 'mean'),
        train_loss_std=('train_loss', 'std'),
        val_loss_mean=('val_loss', 'mean'),
        val_loss_std=('val_loss', 'std'),
    ).reset_index()

    plt.figure(figsize=(10, 6))
    for (opt, lr, bs), g in agg.groupby(['optimizer', 'lr', 'batch_size']):
        label = f"{opt} lr={lr} bs={bs}"
        x = g['epoch']
        y = g['val_loss_mean']
        yerr = g['val_loss_std'].fillna(0)
        plt.plot(x, y, label=label)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.15)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Convergencia de la pérdida (val_loss)')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(outdir, 'loss.png')
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.close()


def plot_accuracy(df, outdir):
    if 'val_acc' not in df.columns:
        print("Column 'val_acc' not found in CSVs; skipping accuracy plot.")
        return
    group_cols = ['optimizer', 'lr', 'batch_size']
    df['lr'] = df['lr'].astype(str)
    df['batch_size'] = df['batch_size'].astype(str)

    agg = df.groupby(group_cols + ['epoch']).agg(
        val_acc_mean=('val_acc', 'mean'),
        val_acc_std=('val_acc', 'std')
    ).reset_index()

    plt.figure(figsize=(10, 6))
    for (opt, lr, bs), g in agg.groupby(['optimizer', 'lr', 'batch_size']):
        label = f"{opt} lr={lr} bs={bs}"
        x = g['epoch']
        y = g['val_acc_mean']
        yerr = g['val_acc_std'].fillna(0)
        plt.plot(x, y, label=label)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.15)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Precisión en validación (val_acc)')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(outdir, 'accuracy.png')
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.close()


def plot_times_boxplot(df, outdir):
    # Tomar elapsed_total_sec de la última epoch por run (archivo origen + seed)
    if 'elapsed_total_sec' not in df.columns:
        print("Column 'elapsed_total_sec' not found; skipping times boxplot.")
        return

    # identificar cada corrida por source file y seed y optimizer
    if 'seed' in df.columns:
        id_cols = ['__source_file', 'seed', 'optimizer', 'lr', 'batch_size']
    else:
        id_cols = ['__source_file', 'optimizer', 'lr', 'batch_size']

    last = df.groupby(id_cols + ['epoch']).tail(1)
    # pero queremos una fila por (source, seed, optimizer)
    last = last.groupby(id_cols).agg(elapsed_total_sec=('elapsed_total_sec', 'max')).reset_index()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='optimizer', y='elapsed_total_sec', data=last)
    sns.swarmplot(x='optimizer', y='elapsed_total_sec', data=last, color='.25')
    plt.ylabel('Tiempo total (s)')
    plt.title('Comparación de tiempos totales por optimizador')
    plt.tight_layout()
    out_path = os.path.join(outdir, 'times_boxplot.png')
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generar gráficas a partir de CSV logs de entrenamiento')
    parser.add_argument('--input', required=True, help='Patrón glob para archivos CSV (ej: "logs\\*.csv")')
    parser.add_argument('--outdir', required=True, help='Directorio de salida para las imágenes')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_csvs(args.input)

    # Asegurar tipos
    if 'epoch' in df.columns:
        df['epoch'] = df['epoch'].astype(int)

    plot_loss(df, args.outdir)
    plot_accuracy(df, args.outdir)
    plot_times_boxplot(df, args.outdir)


if __name__ == '__main__':
    main()
