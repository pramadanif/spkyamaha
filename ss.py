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
    
    return {
        'model': dt_model,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
    }