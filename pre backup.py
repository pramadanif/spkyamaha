import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import datetime
import logging

def preprocess_data(input_file, logger):
    """
    Melakukan pra-pemrosesan data servis motor sesuai dengan persyaratan yang ditentukan
    
    Args:
        input_file (str): Path ke file CSV input
        
    Returns:
        dict: Dictionary yang berisi data yang telah di-preprocessing dan objek preprocessing
    """
    logger.info(f"Membaca data dari: {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Jumlah data awal: {df.shape[0]} baris, {df.shape[1]} kolom")
    logger.info(f"Kolom awal: {df.columns.tolist()}")

    # Langkah 1: Seleksi Fitur (sesuai dengan Tabel 4.1 dalam skripsi)
    selected_features = [
        'Kode Unit Motor',           # 1. Kode Unit Motor
        'Tahun Kendaraan',          # 2. Tahun Kendaraan
        'Kilometer',                # 3. Kilometer
        'Usia_Kendaraan',           # 4. Usia Kendaraan
        'Kondisi_Oli',              # 5. Kondisi Oli
        'Kondisi_Rem',              # 6. Kondisi Rem
        'KM_Terakhir_Ganti_Oli',    # 7. Kilometer Terakhir Ganti Oli
        'Bulan_Terakhir_Ganti_Oli', # 8. Bulan Terakhir Ganti Oli
        'KM_Per_Tahun',             # 9. Kilometer per Tahun
        'Jumlah_Keluhan',           # 10. Keluhan
        'Jns Service'               # Target variable
    ]
    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        logger.warning(f"Peringatan: Fitur berikut tidak ada dalam DataFrame: {missing_features}")
        logger.warning(f"Kolom yang tersedia: {df.columns.tolist()}")
        selected_features = [col for col in selected_features if col in df.columns]
    # Pertahankan hanya fitur yang dipilih
    df = df[selected_features]
    logger.info(f"Jumlah data setelah seleksi fitur: {df.shape[0]} baris")

    # Langkah 2: Penghapusan Data Tidak Relevan (sesuai dengan 4.1.2 dalam skripsi)
    logger.info(f"Jumlah data sebelum filter jenis layanan: {df.shape[0]} baris")
    
    # Tampilkan distribusi jenis service sebelum preprocessing
    logger.info("Distribusi jenis service sebelum preprocessing:")
    logger.info(df['Jns Service'].value_counts())
    
    # Normalisasi dan standarisasi jenis service
    df['Jns Service'] = df['Jns Service'].str.strip().str.lower()
    
    def standardize_service(service):
        if pd.isna(service):
            return 'other'
        
        service = str(service).strip().lower()

        # Pola untuk kupon gratis yang akan digolongkan sebagai 'other'
        kupon_gratis_patterns = [
            'kupon gratis', 'gratis', 'kupon', 'free service', 'complimentary'
        ]

        # Pola untuk repair/perbaikan
        repair_patterns = [
            'repair', 'perbaikan', 'service perbaikan', 'maintenance repair'
        ]
        
        # Pola untuk rutin/berkala
        rutin_patterns = [
            'rutin', 'berkala', 'service rutin', 'maintenance rutin', 'perawatan rutin'
        ]

        # Cek pola kupon gratis -> other
        if any(pattern in service for pattern in kupon_gratis_patterns):
            return 'other'
        # Cek pola repair
        elif any(pattern in service for pattern in repair_patterns):
            return 'repair'
        # Cek pola rutin
        elif any(pattern in service for pattern in rutin_patterns):
            return 'rutin'
        else:
            return 'other'
    
    # Terapkan standardisasi
    df['Jns Service'] = df['Jns Service'].apply(standardize_service)
    
    # Tampilkan distribusi setelah standardisasi
    logger.info("\nDistribusi jenis service setelah standardisasi:")
    logger.info(df['Jns Service'].value_counts())
    
    # Filter hanya repair dan rutin (sesuai dengan metodologi skripsi)
    n_before_service = df.shape[0]
    df = df[df['Jns Service'].isin(['repair', 'rutin'])]
    n_after_service = df.shape[0]
    
    logger.info(f"\nJumlah data setelah penghapusan data tidak relevan: {n_after_service} baris")
    logger.info(f"Data yang dihapus: {n_before_service-n_after_service} baris")
    
    # Tampilkan distribusi final
    logger.info("Distribusi final jenis service:")
    logger.info(df['Jns Service'].value_counts())

    # Simpan data yang sudah bersih
    output_cleaned_file = os.path.join(os.path.dirname(input_file), 'indoperkasa2_clean.csv')
    df.to_csv(output_cleaned_file, index=False)
    logger.info(f"Data hasil preprocessing disimpan ke: {output_cleaned_file}")
    
    # Langkah 3: Transformasi Data Kategorikal dengan Label Encoding (sesuai dengan 4.1.3 dalam skripsi)
    logger.info("\n=== TRANSFORMASI DATA KATEGORIKAL ===")
    logger.info("Menerapkan Label Encoding pada fitur kategorikal:")
    
    categorical_features = ['Kode Unit Motor', 'Kondisi_Oli', 'Kondisi_Rem']
    modes = {}
    for col in categorical_features:
        modes[col] = df[col].mode()[0]

    label_encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        # Simpan kategori sebelum encoding
        unique_cats_before = df[col].unique()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
        # Tampilkan hasil encoding
        logger.info(f"\n  {col}:")
        logger.info(f"    Kategori unik sebelum encoding: {unique_cats_before.tolist()}")
        logger.info(f"    Hasil encoding:")
        # Tampilkan mapping dari kategori ke nilai encode
        for cat, encoded_val in zip(le.classes_, le.transform(le.classes_)):
            logger.info(f"      '{cat}' -> {encoded_val}")
    
    # Terapkan Label Encoding pada target variable
    logger.info("\nMenerapkan Label Encoding pada target variable (Jns Service):")
    target_encoder = LabelEncoder()
    target_cats_before = df['Jns Service'].unique()
    df['Jns Service'] = target_encoder.fit_transform(df['Jns Service'])
    logger.info(f"  Kategori target sebelum encoding: {target_cats_before.tolist()}")
    logger.info(f"  Hasil encoding target:")
    for cat, encoded_val in zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)):
        logger.info(f"    '{cat}' -> {encoded_val}")
        
    logger.info("\nTransformasi data kategorikal selesai.")
    
    # Langkah 4: Pembagian Dataset (sesuai dengan 4.1.4 dalam skripsi)
    logger.info("\n=== PEMBAGIAN DATASET ===")
    feature_list = [col for col in df.columns if col != 'Jns Service']
    X = df[feature_list]
    y = df['Jns Service']
    
    logger.info(f"Total fitur yang digunakan: {len(feature_list)}")
    logger.info(f"Nama fitur: {feature_list}")
    logger.info(f"Dimensi data fitur (X): {X.shape}")
    logger.info(f"Dimensi data target (y): {y.shape[0]}")
    
    # Tampilkan distribusi sebelum pembagian
    logger.info("\nDistribusi target sebelum pembagian:")
    for val, label in zip(target_encoder.transform(target_encoder.classes_), target_encoder.classes_):
        logger.info(f"  {label} (encoded: {val}): {sum(y == val)} data")
    
    # Pembagian dataset: 80% training, 20% testing (sesuai dengan metodologi skripsi)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info("\nHasil pembagian dataset:")
    logger.info(f"  Data Latih (Training Set): {X_train.shape[0]} data ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"  Data Uji (Testing Set): {X_test.shape[0]} data ({len(X_test)/len(X)*100:.1f}%)")
    
    # Tampilkan distribusi setelah pembagian
    logger.info("\nDistribusi target pada Data Latih:")
    for val, label in zip(target_encoder.transform(target_encoder.classes_), target_encoder.classes_):
        logger.info(f"  {label}: {sum(y_train == val)} data")
        
    logger.info("\nDistribusi target pada Data Uji:")
    for val, label in zip(target_encoder.transform(target_encoder.classes_), target_encoder.classes_):
        logger.info(f"  {label}: {sum(y_test == val)} data")
        
    logger.info("\nPembagian dataset menggunakan random splitting dengan stratifikasi selesai.")
    
    # Kembalikan data yang telah di-preprocessing dan objek preprocessing
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_list': feature_list,
        'modes': modes,
    }

def train_decision_tree(preprocessed_data, logger):
    """
    Melatih pengklasifikasi Decision Tree menggunakan data yang telah di-preprocessing
    
    Args:
        preprocessed_data (dict): Dictionary yang berisi data yang telah di-preprocessing
        
    Returns:
        tuple: Model terlatih dan metrik performa
    """
    X_train = preprocessed_data['X_train']
    y_train = preprocessed_data['y_train']
    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']
    
    logger.info(f"Melatih Decision Tree dengan {len(X_train)} sampel training...")
    
    # Inisialisasi dan latih model Decision Tree
    dt_model = DecisionTreeClassifier(
        criterion='gini', 
        max_depth=10, 
        min_samples_leaf=5, 
        random_state=42
    )
    
    dt_model.fit(X_train, y_train)
    
    # Buat prediksi pada data uji
    y_pred = dt_model.predict(X_test)
    
    # Hitung metrik performa
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info(f"\nPerforma Model:")
    logger.info(f"Akurasi: {accuracy:.4f}")
    logger.info(f"\nLaporan Klasifikasi:\n{class_report}")
    logger.info(f"\nMatriks Konfusi:\n{conf_matrix}")
    
    # Hitung kepentingan fitur
    feature_importance = pd.DataFrame({
        'Fitur': preprocessed_data['feature_list'],
        'Tingkat_Kepentingan': dt_model.feature_importances_
    }).sort_values('Tingkat_Kepentingan', ascending=False)
    
    logger.info(f"\nTingkat Kepentingan Fitur:\n{feature_importance}")
    
    return {
        'model': dt_model,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance
    }

def save_model_data(model_results, preprocessed_data, output_dir, logger):
    """
    Menyimpan model, objek preprocessing, dan hasil
    
    Args:
        model_results (dict): Dictionary yang berisi model dan metrik performa
        preprocessed_data (dict): Dictionary yang berisi objek preprocessing
        output_dir (str): Direktori untuk menyimpan hasil
    """
    import pickle
    from datetime import datetime
    
    # Buat direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Simpan model
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model_results['model'], f)
    
    # Simpan objek preprocessing
    with open(os.path.join(output_dir, 'preprocessing.pkl'), 'wb') as f:
        preprocessing_objects = {
            'label_encoders': preprocessed_data['label_encoders'],
            'target_encoder': preprocessed_data['target_encoder'],
            'feature_list': preprocessed_data['feature_list'],
            'modes': preprocessed_data['modes']
        }
        pickle.dump(preprocessing_objects, f)
    
    # Simpan data train dan test untuk evaluasi (preprocessed_data.pkl)
    with open(os.path.join(output_dir, 'preprocessed_data.pkl'), 'wb') as f:
        test_data = {
            'X_test': preprocessed_data['X_test'],
            'y_test': preprocessed_data['y_test'],
            'X_train': preprocessed_data['X_train'],
            'y_train': preprocessed_data['y_train']
        }
        pickle.dump(test_data, f)
    
    # Simpan data train dan test sebagai CSV
    X_train = preprocessed_data['X_train']
    y_train = preprocessed_data['y_train']
    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']
    
    # Gabungkan X dan y untuk train
    train_data = X_train.copy()
    train_data['Jns Service'] = y_train
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    
    # Gabungkan X dan y untuk test
    test_data = X_test.copy()
    test_data['Jns Service'] = y_test
    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Simpan metrik performa sebagai teks
    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Hasil Pelatihan Model - {current_time}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Akurasi Model: {model_results['accuracy']:.4f}\n\n")
        f.write(f"Laporan Klasifikasi:\n{model_results['classification_report']}\n\n")
        f.write(f"Matriks Konfusi:\n{model_results['confusion_matrix']}\n\n")
        
        f.write("Tingkat Kepentingan Fitur:\n")
        for idx, row in model_results['feature_importance'].iterrows():
            f.write(f"{row['Fitur']}: {row['Tingkat_Kepentingan']:.6f}\n")
    
    # Simpan tingkat kepentingan fitur sebagai CSV
    model_results['feature_importance'].to_csv(
        os.path.join(output_dir, 'feature_importance.csv'),
        index=False
    )
    
    logger.info(f"\nModel dan data preprocessing disimpan ke: {output_dir}")
    logger.info("File yang disimpan:")
    logger.info(f"  - model.pkl (model terlatih)")
    logger.info(f"  - preprocessing.pkl (objek preprocessing)")
    logger.info(f"  - preprocessed_data.pkl (data train/test dalam format pickle)")
    logger.info(f"  - train.csv (data training dalam format CSV)")
    logger.info(f"  - test.csv (data testing dalam format CSV)")
    logger.info(f"  - performance_metrics.txt (metrik performa)")
    logger.info(f"  - feature_importance.csv (kepentingan fitur)")

if __name__ == "__main__":
    # Tentukan jalur input dan output
    input_file = os.path.join(os.path.dirname(__file__), 'indoperkasaff.csv')
    output_dir = os.path.join(os.path.dirname(__file__), 'model_output')

    # Buat direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Konfigurasi logging
    log_file = os.path.join(output_dir, 'preprocessing_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # Menyimpan ke file
            logging.StreamHandler() # Menampilkan di konsol
        ]
    )
    logger = logging.getLogger()

    try:
        # Pra-pemrosesan data
        logger.info("Memulai pra-pemrosesan data...")
        preprocessed_data = preprocess_data(input_file, logger)
        
        # Melatih model
        logger.info("\nMelatih model decision tree...")
        model_results = train_decision_tree(preprocessed_data, logger)
        
        # Menyimpan model dan data
        logger.info("\nMenyimpan model dan hasil...")
        save_model_data(model_results, preprocessed_data, output_dir, logger)
        
        logger.info("\nPra-pemrosesan dan pelatihan model selesai dengan sukses.")
        
    except Exception as e:
        logger.error(f"Terjadi error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())