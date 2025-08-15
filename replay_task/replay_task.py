import cv2
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import os

class ReplayLogoDetector:
    def __init__(self, logo_template: np.ndarray = None, threshold_value: int = 127):
        """
        Initialize the replay logo detector.
        
        Args:
            logo_template: Reference logo image for histogram comparison
            threshold_value: Threshold value for binary conversion (0-255)
        """
        self.logo_template = logo_template
        self.threshold_value = threshold_value
        self.logo_histogram = None
        
        # Calculate logo histogram if template is provided
        if logo_template is not None:
            self.logo_histogram = self._calculate_histogram(logo_template)
    
    def set_logo_template(self, logo_template: np.ndarray):
        """Set or update the logo template and calculate its histogram."""
        self.logo_template = logo_template
        self.logo_histogram = self._calculate_histogram(logo_template)
    
    def _calculate_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate histogram for an image.
        
        Args:
            image: Input image (can be color or grayscale)
            
        Returns:
            Normalized histogram array
        """
        if len(image.shape) == 3:
            # Convert to grayscale for histogram calculation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # Normalize histogram
        hist = hist.flatten() / np.sum(hist)
        return hist
    
    def _threshold_based_detection(self, frame: np.ndarray, 
                                  white_pixel_threshold: float = 0.3) -> bool:
        """
        Threshold-based logo detection method.
        
        Args:
            frame: Input frame
            white_pixel_threshold: Minimum ratio of white pixels to detect logo
            
        Returns:
            True if logo is detected, False otherwise
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame


        #gray = cv2.equalizeHist(gray)
        # binary = cv2.adaptiveThreshold(
        # gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        # cv2.THRESH_BINARY_INV, 11, 2
        # )
        #gray = cv2.equalizeHist(gray)
        # Apply threshold to create binary image
        _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Count white pixels
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.shape[0] * binary.shape[1]
        white_ratio = white_pixels / total_pixels
        
        return white_ratio > white_pixel_threshold, binary
    
    def _histogram_based_detection(self, frame: np.ndarray, 
                                 similarity_threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Histogram-based logo detection method.
        
        Args:
            frame: Input frame
            similarity_threshold: Minimum correlation coefficient for detection
            
        Returns:
            Tuple of (detection_result, similarity_score)
        """
        if self.logo_histogram is None:
            raise ValueError("Logo template not set. Use set_logo_template() first.")
        
        # Calculate frame histogram
        frame_histogram = self._calculate_histogram(frame)
        
        # Compare histograms using correlation coefficient
        correlation = cv2.compareHist(
            self.logo_histogram.astype(np.float32),
            frame_histogram.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        
        return correlation > similarity_threshold, correlation
    
    def detect_logo(self, frame: np.ndarray, 
                   method: str = 'histogram',
                   **kwargs) -> dict:
        """
        Detect replay logo in a frame using specified method.
        
        Args:
            frame: Input video frame
            method: Detection method ('threshold', 'histogram', or 'combined')
            **kwargs: Additional parameters for detection methods
            
        Returns:
            Dictionary with detection results
        """
        results = {
            'frame_shape': frame.shape,
            'method': method,
            'logo_detected': False,
            'confidence': 0.0,
            'binary': frame.shape,
        }
        
        if method == 'threshold':
            threshold_param = kwargs.get('white_pixel_threshold', 0.5)
            results['logo_detected'], results['binary'] = self._threshold_based_detection(frame, threshold_param)
            results['confidence'] = 1.0 if results['logo_detected'] else 0.0
            
        elif method == 'histogram':
            if self.logo_histogram is None:
                raise ValueError("Logo template required for histogram method")
            
            similarity_threshold = kwargs.get('similarity_threshold', 0.8)
            detected, similarity = self._histogram_based_detection(frame, similarity_threshold)
            results['logo_detected'] = detected
            results['confidence'] = similarity
            
        elif method == 'combined':
            # Use both methods - logo detected if either method succeeds
            threshold_detected, results['binary'] = self._threshold_based_detection(
                frame, kwargs.get('white_pixel_threshold', 0.3)
            )
            
            if self.logo_histogram is not None:
                hist_detected, similarity = self._histogram_based_detection(
                    frame, kwargs.get('similarity_threshold', 0.7)
                )
                results['logo_detected'] = threshold_detected or hist_detected
                results['confidence'] = similarity if hist_detected else (0.5 if threshold_detected else 0.0)
            else:
                results['logo_detected'] = threshold_detected
                results['confidence'] = 1.0 if threshold_detected else 0.0
        
        else:
            raise ValueError("Method must be 'threshold', 'histogram', or 'combined'")
        
        return results
    
    def process_video(self, video_path: str, 
                     method: str = 'histogram',
                     display_results: bool = True,
                     **kwargs) -> List[dict]:
        """
        Process entire video for logo detection.
        
        Args:
            video_path: Path to video file
            method: Detection method to use
            display_results: Whether to display detection results
            **kwargs: Additional parameters for detection methods
            
        Returns:
            List of detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            

            # Detect logo in current frame
            result = self.detect_logo(frame, method, **kwargs)
            result['frame_number'] = frame_count
            results.append(result)
            
            # Display results if requested
            if display_results:
                status = "LOGO DETECTED" if result['logo_detected'] else "NO LOGO"
                confidence = result['confidence']
                
                # Add text overlay
                #frame= result['binary']
                cv2.putText(frame, f"{status} (Conf: {confidence:.2f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 0) if result['logo_detected'] else (0, 0, 255), 2)
                
                cv2.imshow('Logo Detection', frame)
                
                # Press 'q' to quit
                if result['logo_detected']:
                    print("Pausing... press any key to continue")
                    key = cv2.waitKey(0)  # Wait indefinitely for a key press
                    if key & 0xFF == ord('q'):  # Quit if 'q' pressed
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        return results
    
    def analyze_detection_results(self, results: List[dict]) -> dict:
        """
        Analyze detection results and provide statistics.
        
        Args:
            results: List of detection results from process_video()
            
        Returns:
            Dictionary with analysis statistics
        """
        total_frames = len(results)
        detected_frames = sum(1 for r in results if r['logo_detected'])
        detection_rate = detected_frames / total_frames if total_frames > 0 else 0
        
        confidences = [r['confidence'] for r in results if r['logo_detected']]
        avg_confidence = np.mean(confidences) if confidences else 0

        frames_logo = [r['frame_number'] for r in results if r['logo_detected'] ]
        


        analysis = {
            'total_frames': total_frames,
            'frames_with_logo': detected_frames,
            'detection_rate': detection_rate,
            'average_confidence': avg_confidence,
            'confidence_std': np.std(confidences) if confidences else 0,
            'frames_logo': frames_logo
        }
        
        return analysis




if __name__ == "__main__":
    
    #path = "results\\output_video_segment\\2015-05-17 - 20-00 Espanyol 1 - 4 Real Madrid clip_1464.0_1506.0.mp4"
    path =  "results\output_video_segment\\2016-11-26 - 17-30 Eintracht Frankfurt 2 - 1 Dortmundclip_966.0000000000001_1050.0.mp4"
    #path = 'results\output_video_segment\\2016-11-26 - 17-30 Eintracht Frankfurt 2 - 1 Dortmund clip_1692.0_1725.0.mp4'
    #path = 'results\output_video_segment\\2016-12-19 - 23-00 Everton 0 - 1 Liverpool clip_2922.0_3000.0.mp4'
    #path = 'results\output_video_segment\\2015-08-23 - 15-30 West Brom 2 - 3 Chelsea_clip_1464.0_1554.0.mp4'
    
    logo_path = 'replay_task/logos/goal.png'
    detector = ReplayLogoDetector()
    # results = detector.process_video(path, method='threshold')
    # analysis = detector.analyze_detection_results(results)
    # print(analysis)

    folder_data= "results/output_video_segment"
    list_dir = os.listdir(folder_data)

    for video in list_dir:
        path = f'results/output_video_segment/{video}'
        print(f'Processing...{path}')
        if os.path.exists(logo_path):
            logo = cv2.imread(logo_path)
            detector.set_logo_template(logo)
        else:
            print('Logo not found')
        
        results = detector.process_video(path, method='histogram', display_results=True)  
        analysis = detector.analyze_detection_results(results)
        print(analysis)
        #todo conver frames to milisecond, eliminate or combine frames that are close, create diuctionary
        