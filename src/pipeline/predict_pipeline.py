import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.threshold = 0.55  # Threshold for Alive/Dead classification
        
    def load_artifacts(self):
        """Load model and preprocessor"""
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            self.model = load_object(file_path=model_path)
            self.preprocessor = load_object(file_path=preprocessor_path)
            
            return self.model, self.preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        """
        Make prediction for breast cancer survival
        
        Args:
            features: DataFrame with processed features (ready for preprocessor)
        
        Returns:
            predictions: array of predictions (1=Alive, 0=Dead)
        """
        try:
            if self.model is None or self.preprocessor is None:
                self.load_artifacts()
            
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_with_proba(self, features):
        """Make prediction with probability scores"""
        try:
            if self.model is None or self.preprocessor is None:
                self.load_artifacts()
            
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            
            proba = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(data_scaled)
            
            return preds, proba
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Age: int,
                 Gender: str,
                 Protein1: float,
                 Protein2: float,
                 Protein3: float,
                 Protein4: float,
                 Tumour_Stage: str,
                 Histology: str,
                 HER2_status: str,
                 Surgery_type: str,
                 Date_of_Surgery: str = None,
                 Date_of_Last_Visit: str = None):
        
        self.Age = Age
        self.Gender = Gender
        self.Protein1 = Protein1
        self.Protein2 = Protein2
        self.Protein3 = Protein3
        self.Protein4 = Protein4
        self.Tumour_Stage = Tumour_Stage
        self.Histology = Histology
        self.HER2_status = HER2_status
        self.Surgery_type = Surgery_type
        self.Date_of_Surgery = Date_of_Surgery
        self.Date_of_Last_Visit = Date_of_Last_Visit
    
    def _calculate_survived_days(self):
        """Calculate Survived_days from dates"""
        try:
            if self.Date_of_Surgery and self.Date_of_Last_Visit:
                surgery_date = pd.to_datetime(self.Date_of_Surgery, format='%d-%b-%y', errors='coerce')
                last_visit_date = pd.to_datetime(self.Date_of_Last_Visit, format='%d-%b-%y', errors='coerce')
                
                if pd.notna(surgery_date) and pd.notna(last_visit_date):
                    survived_days = (last_visit_date - surgery_date).days
                    if survived_days > 0:
                        return survived_days
        except:
            pass
        # Return median value from training
        return 365
    
    def _encode_tumour_stage(self):
        """Convert tumour stage to encoded value"""
        stage_map = {'I': 1, 'II': 2, 'III': 3}
        return stage_map.get(self.Tumour_Stage, 2)
    
    def get_data_as_dataframe(self):
        """
        Convert input data to DataFrame format expected by preprocessor.
        This matches the exact columns from data_transformation.py
        """
        try:
            # Calculate derived features
            survived_days = self._calculate_survived_days()
            tumour_stage_encoded = self._encode_tumour_stage()
            
            # Validate surgery type
            valid_surgeries = ['Lumpectomy', 'Modified Radical Mastectomy', 'Other', 'Simple Mastectomy']
            surgery_type = self.Surgery_type if self.Surgery_type in valid_surgeries else 'Other'
            
            # Create DataFrame with exact columns expected by preprocessor
            custom_data_input_dict = {
                "Age": [float(self.Age)],
                "Protein1": [float(self.Protein1)],
                "Protein2": [float(self.Protein2)],
                "Protein3": [float(self.Protein3)],
                "Protein4": [float(self.Protein4)],
                "Survived_days": [survived_days],
                "Tumour_Stage_Encoded": [tumour_stage_encoded],
                "Surgery_type": [surgery_type],
                "Histology": [self.Histology],
                "Gender": [self.Gender],
                "HER2 status": [self.HER2_status]
            }
            
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)


# Quick prediction function
def quick_predict(patient_data_dict):
    """
    Quick prediction function for single patient
    
    Args:
        patient_data_dict: Dictionary with patient information
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Create CustomData object
        custom_data = CustomData(**patient_data_dict)
        
        # Get features DataFrame
        features_df = custom_data.get_data_as_dataframe()
        
        # Make prediction
        pipeline = PredictPipeline()
        predictions, probabilities = pipeline.predict_with_proba(features_df)
        
        result = {
            'prediction': 'Alive' if predictions[0] == 1 else 'Dead',
            'prediction_numeric': int(predictions[0])
        }
        
        if probabilities is not None:
            prob_alive = probabilities[0][1] if len(probabilities[0]) > 1 else probabilities[0][0]
            result['survival_probability'] = round(prob_alive * 100, 1)
            result['death_probability'] = round((1 - prob_alive) * 100, 1)
            
            # Add risk level
            if prob_alive >= 0.7:
                result['risk_level'] = 'Low Risk'
            elif prob_alive >= 0.4:
                result['risk_level'] = 'Medium Risk'
            else:
                result['risk_level'] = 'High Risk'
        
        return result
        
    except Exception as e:
        raise CustomException(e, sys)


# Test the pipeline
if __name__ == "__main__":
    print("="*60)
    print("TESTING BREAST CANCER PREDICTION PIPELINE")
    print("="*60)
    
    # Test patient with good prognosis (should be ALIVE)
    test_patient_alive = {
        'Age': 42,
        'Gender': 'FEMALE',
        'Protein1': 0.95,
        'Protein2': 2.15,
        'Protein3': 0.01,
        'Protein4': -0.05,
        'Tumour_Stage': 'I',
        'Histology': 'Infiltrating Ductal Carcinoma',
        'HER2_status': 'Negative',
        'Surgery_type': 'Lumpectomy',
        'Date_of_Surgery': '20-May-18',
        'Date_of_Last_Visit': '26-Aug-18'
    }
    
    # Test patient with poor prognosis (should be DEAD)
    test_patient_dead = {
        'Age': 85,
        'Gender': 'MALE',
        'Protein1': -1.50,
        'Protein2': -0.80,
        'Protein3': -1.00,
        'Protein4': -1.20,
        'Tumour_Stage': 'III',
        'Histology': 'Infiltrating Ductal Carcinoma',
        'HER2_status': 'Positive',
        'Surgery_type': 'Modified Radical Mastectomy',
        'Date_of_Surgery': '20-May-18',
        'Date_of_Last_Visit': '26-Aug-18'
    }
    
    try:
        print("\n Test 1: Good Prognosis Patient")
        
        result1 = quick_predict(test_patient_alive)
        for key, value in result1.items():
            print(f"   {key}: {value}")
        
        print("\n Test 2: Poor Prognosis Patient")
        
        result2 = quick_predict(test_patient_dead)
        for key, value in result2.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")
        