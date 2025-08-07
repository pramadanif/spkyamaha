import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def save_confusion_matrix_image(cm, class_names, file_path):
    """Generates and saves a confusion matrix plot."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close() # Close the plot to prevent it from displaying
        return True, None
    except Exception as e:
        return False, str(e)


def evaluate_model(model_dir, log_file):
    """
    Mengevaluasi model yang telah dilatih menggunakan data uji yang telah diproses sebelumnya.

    Args:
        model_dir (str): Direktori tempat model dan data disimpan.
        log_file (str): Path untuk menyimpan log evaluasi.
    """
    # Konfigurasi logging untuk menyimpan ke file dan menampilkan di konsol
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    try:
        # 1. Muat Model dan Data yang Telah Diproses Sebelumnya
        logger.info(f"Membaca model dan data dari direktori: {model_dir}")

        # Muat model
        model_path = os.path.join(model_dir, 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("✓ Model berhasil dimuat")

        # Muat data test
        data_path = os.path.join(model_dir, 'preprocessed_data.pkl')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data preprocessing tidak ditemukan: {data_path}")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        X_test = data['X_test']
        y_test = data['y_test']
        logger.info("✓ Data test berhasil dimuat")

        # Muat objek preprocessing
        preprocessing_path = os.path.join(model_dir, 'preprocessing.pkl')
        if not os.path.exists(preprocessing_path):
            raise FileNotFoundError(f"Objek preprocessing tidak ditemukan: {preprocessing_path}")
        
        with open(preprocessing_path, 'rb') as f:
            preprocessing_objects = pickle.load(f)
        target_encoder = preprocessing_objects['target_encoder']
        class_names = target_encoder.classes_
        logger.info("✓ Objek preprocessing berhasil dimuat")

        logger.info(f"\nInformasi Dataset:")
        logger.info(f"Jumlah data test: {len(X_test)} sampel")
        logger.info(f"Jumlah fitur: {X_test.shape[1]}")
        logger.info(f"Jumlah kelas: {len(class_names)} ({', '.join(class_names)})")

        # 2. Lakukan Prediksi
        logger.info("\nMelakukan prediksi pada data test...")
        y_pred = model.predict(X_test)
        logger.info("✓ Prediksi selesai")

        # 3. Hitung Metrik yang Diperlukan
        logger.info("\nMenghitung metrik evaluasi...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Classification report untuk detail per kelas
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        # 4. Format dan Catat Hasil dalam Bentuk Tabel
        logger.info("\n" + "="*60)
        logger.info("                   HASIL EVALUASI MODEL")
        logger.info("="*60)

        # Tabel Metrik Performa Keseluruhan
        logger.info("\n**1. METRIK PERFORMA KESELURUHAN**")
        logger.info("-" * 40)
        logger.info("| Metric            | Score     |")
        logger.info("|-------------------|-----------|")
        logger.info(f"| Accuracy          | {accuracy:.4f}    |")
        logger.info(f"| Precision (macro) | {precision:.4f}    |")
        logger.info(f"| Recall (macro)    | {recall:.4f}    |")
        logger.info(f"| F1-Score (macro)  | {f1:.4f}    |")
        logger.info("-" * 40)

        # Tabel Metrik Per Kelas
        logger.info("\n**2. METRIK PERFORMA PER KELAS**")
        logger.info("-" * 60)
        logger.info("| Kelas   | Precision | Recall | F1-Score | Support |")
        logger.info("|---------|-----------|--------|----------|---------|")
        
        for class_name in class_names:
            if class_name in class_report:
                precision_class = class_report[class_name]['precision']
                recall_class = class_report[class_name]['recall']
                f1_class = class_report[class_name]['f1-score']
                support_class = int(class_report[class_name]['support'])
                
                logger.info(f"| {class_name:<7} | {precision_class:>9.4f} | {recall_class:>6.4f} | {f1_class:>8.4f} | {support_class:>7} |")
        
        logger.info("-" * 60)

        # Tabel Confusion Matrix
        logger.info("\n**3. CONFUSION MATRIX**")
        logger.info("-" * 40)
        
        # Header confusion matrix
        header_parts = ["Actual\\Predicted"] + [f"{name}" for name in class_names]
        max_width = max(len(part) for part in header_parts + [str(cm.max())])
        max_width = max(max_width, 10)  # minimum width
        
        # Print header
        header_line = "|"
        separator_line = "|"
        for part in header_parts:
            header_line += f" {part:^{max_width}} |"
            separator_line += "-" * (max_width + 2) + "|"
        
        logger.info(header_line)
        logger.info(separator_line)

        # Print matrix rows
        for i, class_name in enumerate(class_names):
            row_line = f"| {class_name:^{max_width}} |"
            for j in range(len(class_names)):
                row_line += f" {cm[i][j]:^{max_width}} |"
            logger.info(row_line)
        
        logger.info(separator_line)

        # Tambahan informasi interpretasi
        logger.info("\n**4. INTERPRETASI HASIL**")
        logger.info("-" * 40)
        
        # Hitung total prediksi benar dan salah
        total_correct = np.trace(cm)
        total_samples = np.sum(cm)
        total_incorrect = total_samples - total_correct
        
        logger.info(f"Total sampel test     : {total_samples}")
        logger.info(f"Prediksi benar        : {total_correct} ({total_correct/total_samples*100:.1f}%)")
        logger.info(f"Prediksi salah        : {total_incorrect} ({total_incorrect/total_samples*100:.1f}%)")
        
        # Analisis per kelas
        logger.info("\nAnalisis per kelas:")
        for i, class_name in enumerate(class_names):
            true_positive = cm[i][i]
            total_actual = np.sum(cm[i, :])
            total_predicted = np.sum(cm[:, i])
            
            logger.info(f"- {class_name}:")
            logger.info(f"  * Aktual: {total_actual}, Diprediksi: {total_predicted}")
            logger.info(f"  * Benar diprediksi: {true_positive}")
            if total_actual > 0:
                logger.info(f"  * Akurasi kelas: {true_positive/total_actual*100:.1f}%")

        # 5. Simpan hasil ke file terpisah
        logger.info("\nMenyimpan hasil evaluasi...")
        evaluation_results = {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'class_names': class_names
        }
        
        # Simpan hasil evaluasi sebagai pickle
        results_path = os.path.join(model_dir, 'evaluation_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(evaluation_results, f)
        
        # Simpan confusion matrix sebagai CSV
        cm_df = pd.DataFrame(cm, 
                           index=[f"Actual_{name}" for name in class_names],
                           columns=[f"Predicted_{name}" for name in class_names])
        cm_csv_path = os.path.join(model_dir, 'confusion_matrix.csv')
        cm_df.to_csv(cm_csv_path)
        
        # Simpan metrik per kelas sebagai CSV
        metrics_per_class = []
        for class_name in class_names:
            if class_name in class_report:
                metrics_per_class.append({
                    'Class': class_name,
                    'Precision': class_report[class_name]['precision'],
                    'Recall': class_report[class_name]['recall'],
                    'F1-Score': class_report[class_name]['f1-score'],
                    'Support': class_report[class_name]['support']
                })
        
        metrics_df = pd.DataFrame(metrics_per_class)
        metrics_csv_path = os.path.join(model_dir, 'metrics_per_class.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)

        # 6. Generate and save confusion matrix image
        cm_image_path = os.path.join(model_dir, 'confusion_matrix.png')
        success, error_msg = save_confusion_matrix_image(cm, class_names, cm_image_path)

        logger.info("\n" + "="*60)
        logger.info("                    FILE OUTPUT")
        logger.info("="*60)
        logger.info(f"Log evaluasi lengkap  : {log_file}")
        logger.info(f"Hasil evaluasi (pkl)  : {results_path}")
        logger.info(f"Confusion matrix (csv): {cm_csv_path}")
        if success:
            logger.info(f"Confusion matrix (png): {cm_image_path}")
        else:
            logger.warning(f"⚠️ Gambar confusion matrix tidak tersimpan: {error_msg}")
        logger.info(f"Metrik per kelas (csv): {metrics_csv_path}")
        logger.info("="*60)

        if not success:
             logger.warning("\nTips: Pastikan library 'matplotlib' dan 'seaborn' sudah terinstall.")
             logger.warning("Jalankan: pip install matplotlib seaborn")

        logger.info("\n✓ Evaluasi model selesai dengan sukses!")

    except FileNotFoundError as e:
        logger.error(f"❌ ERROR: File tidak ditemukan.")
        logger.error(f"Detail: {e}")
        logger.error("\nPastikan file berikut sudah ada (jalankan preprocessing terlebih dahulu):")
        logger.error(f"- {os.path.join(model_dir, 'model.pkl')}")
        logger.error(f"- {os.path.join(model_dir, 'preprocessed_data.pkl')}")
        logger.error(f"- {os.path.join(model_dir, 'preprocessing.pkl')}")
    except Exception as e:
        logger.error(f"❌ Terjadi error saat evaluasi: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Tentukan direktori input/output
    base_dir = os.path.dirname(__file__)
    model_output_dir = os.path.join(base_dir, 'model_output')
    
    # Tentukan file log output
    log_output_file = os.path.join(model_output_dir, 'evaluation_log.txt')

    # Buat direktori output jika belum ada
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    print("Memulai evaluasi model...")
    print(f"📁 Direktori model: {model_output_dir}")
    print(f"📄 Log output: {log_output_file}")
    print("-" * 50)

    # Jalankan fungsi evaluasi
    evaluate_model(model_output_dir, log_output_file)