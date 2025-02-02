"""
Hand Gesture Recognition with Machine Learning
Created by: Aakash Yadav
"""

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
import os
import customtkinter as ctk
from threading import Thread
import time
import pyautogui

class HandGestureML:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_data = []
        self.gesture_labels = []
        self.model = None
        self.recording = False
        self.current_gesture = "None"
        self.algorithms = {
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'SVM': SVC(kernel='rbf', probability=True),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50))
        }
        
        self.good_gestures = {
            "thumbs_up": "👍 Good - System Control",
            "peace": "✌️ Good - Volume Up",
            "palm": "✋ Good - Navigation",
            "fist": "👊 Good - Selection",
            "ok_sign": "👌 Good - Confirmation"
        }
        
        self.bad_gestures = {
            "middle_finger": "🖕 Bad - Inappropriate",
            "gun": "🔫 Bad - Violent gesture",
            "thumbs_down": "👎 Bad - Negative gesture"
        }
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        self.root = ctk.CTk()
        self.root.title("Hand Gesture ML System - Created by Aakash Yadav")
        self.root.geometry("1200x900")
        
        # Main frame with two columns
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Left column for controls
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=10)
        
        # Right column for visualizations
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=10)
        
        self.create_control_panel(left_frame)
        self.create_visualization_panel(right_frame)
        
        # Add creator label
        creator_label = ctk.CTkLabel(
            main_frame,
            text="Created by: Aakash Yadav",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        creator_label.pack(pady=5)
        
    def create_control_panel(self, parent):
        # Training Controls
        training_frame = ctk.CTkFrame(parent)
        training_frame.pack(pady=10, fill="x")
        
        self.gesture_name = ctk.CTkEntry(training_frame, placeholder_text="Enter gesture name")
        self.gesture_name.pack(pady=5)
        
        self.record_btn = ctk.CTkButton(
            training_frame, 
            text="Record Gesture", 
            command=self.toggle_recording
        )
        self.record_btn.pack(pady=5)
        
        # Algorithm Selection
        algo_frame = ctk.CTkFrame(parent)
        algo_frame.pack(pady=10, fill="x")
        
        ctk.CTkLabel(algo_frame, text="Select Algorithm:").pack()
        self.algo_var = ctk.StringVar(value="Random Forest")
        for algo in self.algorithms.keys():
            ctk.CTkRadioButton(algo_frame, text=algo, variable=self.algo_var, value=algo).pack()
        
        # Training Button
        self.train_btn = ctk.CTkButton(
            parent, 
            text="Train Model", 
            command=self.train_model
        )
        self.train_btn.pack(pady=10)
        
        # Status Labels
        self.status_label = ctk.CTkLabel(parent, text="Status: Ready")
        self.status_label.pack(pady=5)
        
        self.gesture_label = ctk.CTkLabel(parent, text="Current Gesture: None")
        self.gesture_label.pack(pady=5)
        
        self.dataset_label = ctk.CTkLabel(parent, text="Dataset Size: 0 samples")
        self.dataset_label.pack(pady=5)
        
    def create_visualization_panel(self, parent):
        self.viz_frame = ctk.CTkFrame(parent)
        self.viz_frame.pack(fill="both", expand=True)
        
        # Create matplotlib figure for visualizations
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(6, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add gesture feedback panel
        self.feedback_frame = ctk.CTkFrame(parent)
        self.feedback_frame.pack(fill="x", pady=10)
        
        self.gesture_feedback = ctk.CTkLabel(
            self.feedback_frame,
            text="Gesture Feedback",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.gesture_feedback.pack(pady=5)
        
    def augment_data(self, features, label):
        augmented_features = [features]
        augmented_labels = [label]
        
        # Add noise
        noise_feature = features + np.random.normal(0, 0.01, len(features))
        augmented_features.append(noise_feature)
        augmented_labels.append(label)
        
        # Scale
        scale_feature = features * np.random.uniform(0.95, 1.05, len(features))
        augmented_features.append(scale_feature)
        augmented_labels.append(label)
        
        # Rotate (for 3D points)
        rotated_feature = self.rotate_features(features)
        augmented_features.append(rotated_feature)
        augmented_labels.append(label)
        
        return augmented_features, augmented_labels
    
    def rotate_features(self, features):
        # Reshape features into (n_points, 3) for rotation
        points = np.array(features).reshape(-1, 3)
        angle = np.random.uniform(-0.1, 0.1)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated_points = np.dot(points, rotation_matrix)
        return rotated_points.flatten()
    
    def update_visualizations(self, y_true, y_pred):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=self.ax1)
        self.ax1.set_title('Confusion Matrix')
        
        # Feature Importance (for Random Forest)
        if isinstance(self.model, RandomForestClassifier):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-10:]
            self.ax2.barh(range(10), importances[indices])
            self.ax2.set_title('Top 10 Important Features')
        
        # Gesture Classification
        self.ax3.clear()
        self.ax3.axis('off')
        gesture_text = "Gesture Classification:\n\nGood Gestures:\n"
        for gesture, desc in self.good_gestures.items():
            gesture_text += f"{desc}\n"
        gesture_text += "\nBad Gestures:\n"
        for gesture, desc in self.bad_gestures.items():
            gesture_text += f"{desc}\n"
        self.ax3.text(0.1, 0.9, gesture_text, fontsize=10, va='top')
        
        self.canvas.draw()
    
    def train_model(self):
        if len(self.gesture_data) < 2:
            self.status_label.configure(text="Need more gesture data to train!")
            return
            
        X = np.array(self.gesture_data)
        y = np.array(self.gesture_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Select and train model
        self.model = self.algorithms[self.algo_var.get()]
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        accuracy = self.model.score(X_test, y_test)
        y_pred = self.model.predict(X_test)
        
        # Update visualizations
        self.update_visualizations(y_test, y_pred)
        
        # Calculate and display metrics
        report = classification_report(y_test, y_pred)
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        metrics_text = f"""
        Model Performance:
        Accuracy: {accuracy:.2f}
        Cross-validation scores: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores)*2:.2f})
        
        Detailed Report:
        {report}
        """
        
        self.status_label.configure(text=metrics_text)
        
        # Save the model
        with open('gesture_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
    
    def extract_features(self, hand_landmarks):
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return features
    
    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.record_btn.configure(text="Stop Recording")
            self.status_label.configure(text=f"Recording gesture: {self.gesture_name.get()}")
        else:
            self.record_btn.configure(text="Record Gesture")
            self.status_label.configure(text="Recording stopped")
    
    def process_frame(self, image, hand_landmarks):
        # Draw landmarks with custom style
        self.mp_draw.draw_landmarks(
            image, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)
        )
        
        # Add gesture feedback text on screen
        features = self.extract_features(hand_landmarks)
        if self.model is not None:
            prediction = self.model.predict([features])[0]
            
            # Get confidence score
            if hasattr(self.model, 'predict_proba'):
                confidence = self.model.predict_proba([features])[0].max() * 100
            else:
                confidence = 0
            
            # Determine if gesture is good or bad
            if prediction in self.good_gestures:
                feedback = f"✅ GOOD: {self.good_gestures[prediction]}"
                color = (0, 255, 0)  # Green
                status = "APPROVED"
            elif prediction in self.bad_gestures:
                feedback = f"❌ BAD: {self.bad_gestures[prediction]}"
                color = (0, 0, 255)  # Red
                status = "REJECTED"
            else:
                feedback = "⚠️ UNKNOWN GESTURE"
                color = (0, 165, 255)  # Orange
                status = "UNKNOWN"
            
            # Add semi-transparent overlay
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (image.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            # Add main feedback text with shadow
            self.add_text_with_shadow(
                image,
                feedback,
                (10, 30),
                1.0,
                color
            )
            
            # Add confidence score
            conf_text = f"Confidence: {confidence:.1f}%"
            self.add_text_with_shadow(
                image,
                conf_text,
                (10, 60),
                0.7,
                (255, 255, 255)
            )
            
            # Add status indicator in top-right corner
            cv2.rectangle(image, 
                        (image.shape[1]-150, 10), 
                        (image.shape[1]-10, 40), 
                        color, 
                        cv2.FILLED)
            self.add_text_with_shadow(
                image,
                status,
                (image.shape[1]-140, 30),
                0.7,
                (255, 255, 255)
            )
            
            # Add hand tracking info
            hand_info = f"Hand Position: ({hand_landmarks.landmark[0].x:.2f}, {hand_landmarks.landmark[0].y:.2f})"
            self.add_text_with_shadow(
                image,
                hand_info,
                (10, image.shape[0] - 50),
                0.6,
                (255, 255, 255)
            )
            
            # Add creator text with fancy styling
            creator_text = "Created by: Aakash Yadav"
            text_size = cv2.getTextSize(
                creator_text, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                2
            )[0]
            
            # Add background for creator text
            cv2.rectangle(
                image,
                (5, image.shape[0] - 30),
                (text_size[0] + 15, image.shape[0] - 5),
                (0, 0, 0),
                cv2.FILLED
            )
            
            self.add_text_with_shadow(
                image,
                creator_text,
                (10, image.shape[0] - 10),
                0.7,
                (255, 255, 255)
            )
            
        return image
    
    def add_text_with_shadow(self, image, text, position, scale, color):
        # Add shadow
        cv2.putText(
            image,
            text,
            (position[0] + 2, position[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )
        # Add main text
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA
        )

    def start_camera(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            success, image = cap.read()
            if not success:
                break
                
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    image = self.process_frame(image, hand_landmarks)
                    
                    if self.recording:
                        features = self.extract_features(hand_landmarks)
                        self.gesture_data.append(features)
                        self.gesture_labels.append(self.gesture_name.get())
                        self.dataset_label.configure(
                            text=f"Dataset Size: {len(self.gesture_data)} samples"
                        )
            
            cv2.imshow('Hand Gesture ML - By Aakash Yadav', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
    
    def perform_action(self, gesture):
        # Map gestures to actions
        actions = {
            "swipe_right": lambda: pyautogui.hotkey('win', 'ctrl', 'right'),
            "swipe_left": lambda: pyautogui.hotkey('win', 'ctrl', 'left'),
            "volume_up": lambda: pyautogui.press('volumeup'),
            "volume_down": lambda: pyautogui.press('volumedown'),
            # Add more gesture-action mappings
        }
        
        if gesture in actions:
            actions[gesture]()
    
    def evaluate_gesture(self, prediction):
        if prediction in self.good_gestures:
            feedback = f"✅ {self.good_gestures[prediction]}"
            color = "green"
        elif prediction in self.bad_gestures:
            feedback = f"❌ {self.bad_gestures[prediction]}"
            color = "red"
        else:
            feedback = "⚠️ Unknown Gesture"
            color = "orange"
            
        self.gesture_feedback.configure(
            text=feedback,
            text_color=color
        )
    
    def run(self):
        camera_thread = Thread(target=self.start_camera)
        camera_thread.start()
        self.root.mainloop()

if __name__ == "__main__":
    app = HandGestureML()
    app.run()