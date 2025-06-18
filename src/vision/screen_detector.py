import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScreenRegion:
    """Represents a detected screen region"""
    corners: np.ndarray  # 4 corner points in order: TL, TR, BR, BL
    width: int
    height: int
    confidence: float
    transform_matrix: Optional[np.ndarray] = None


class ScreenDetector:
    """Detects computer screens in camera images"""
    
    def __init__(self, min_area: int = 10000, aspect_ratio_range: Tuple[float, float] = (1.0, 2.5)):
        self.min_area = min_area
        self.aspect_ratio_range = aspect_ratio_range
        self._last_screen = None
        self._stable_frames = 0
        self._stability_threshold = 5  # Frames before considering detection stable
        
    def detect(self, frame: np.ndarray) -> Optional[ScreenRegion]:
        """
        Detect a screen in the given frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            ScreenRegion if found, None otherwise
        """
        # Try edge-based detection first
        screen = self._detect_by_edges(frame)
        
        if screen is None:
            # Fallback to color-based detection
            screen = self._detect_by_color(frame)
            
        # Check stability
        if screen:
            if self._is_stable_detection(screen):
                self._last_screen = screen
                return screen
                
        return self._last_screen
        
    def _detect_by_edges(self, frame: np.ndarray) -> Optional[ScreenRegion]:
        """Detect screen using edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Canny edge detection
        edges = cv2.Canny(filtered, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find rectangular contours that could be screens
        candidates = []
        for contour in contours:
            # Approximate polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Check if it's a quadrilateral
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > self.min_area:
                    # Check if it's roughly rectangular
                    if self._is_rectangular(approx):
                        candidates.append(approx)
                        
        # Select best candidate
        if candidates:
            # Choose largest area
            best_candidate = max(candidates, key=cv2.contourArea)
            return self._create_screen_region(best_candidate, frame.shape)
            
        return None
        
    def _detect_by_color(self, frame: np.ndarray) -> Optional[ScreenRegion]:
        """Detect screen using color characteristics (bright regions)"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for bright colors (screens are usually bright)
        lower_bright = np.array([0, 0, 100])  # Low saturation, high value
        upper_bright = np.array([180, 100, 255])
        
        # Create mask for bright regions
        mask = cv2.inRange(hsv, lower_bright, upper_bright)
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                # Get bounding rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Check aspect ratio
                width = rect[1][0]
                height = rect[1][1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                        return self._create_screen_region(box, frame.shape)
                        
        return None
        
    def _is_rectangular(self, contour: np.ndarray, angle_tolerance: float = 15) -> bool:
        """Check if a contour is roughly rectangular"""
        if len(contour) != 4:
            return False
            
        # Calculate angles between consecutive edges
        angles = []
        for i in range(4):
            p1 = contour[i][0]
            p2 = contour[(i + 1) % 4][0]
            p3 = contour[(i + 2) % 4][0]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
            angles.append(np.degrees(angle))
            
        # Check if all angles are close to 90 degrees
        return all(abs(angle - 90) < angle_tolerance for angle in angles)
        
    def _create_screen_region(self, corners: np.ndarray, frame_shape: Tuple[int, int]) -> ScreenRegion:
        """Create a ScreenRegion from corner points"""
        # Ensure corners are in correct order (TL, TR, BR, BL)
        corners = self._order_corners(corners)
        
        # Calculate dimensions
        width = int(np.linalg.norm(corners[1] - corners[0]))
        height = int(np.linalg.norm(corners[3] - corners[0]))
        
        # Calculate confidence based on how rectangular it is
        confidence = self._calculate_confidence(corners)
        
        # Calculate perspective transform matrix
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        transform_matrix = cv2.getPerspectiveTransform(
            corners.astype(np.float32),
            dst_points
        )
        
        return ScreenRegion(
            corners=corners,
            width=width,
            height=height,
            confidence=confidence,
            transform_matrix=transform_matrix
        )
        
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as TL, TR, BR, BL"""
        # Reshape if necessary
        if corners.shape[0] == 4 and len(corners.shape) == 3:
            corners = corners.reshape(4, 2)
            
        # Calculate center
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        # Reorder corners
        ordered = corners[sorted_indices]
        
        # Ensure TL is first (lowest sum of coordinates)
        tl_idx = np.argmin(ordered[:, 0] + ordered[:, 1])
        ordered = np.roll(ordered, -tl_idx, axis=0)
        
        return ordered
        
    def _calculate_confidence(self, corners: np.ndarray) -> float:
        """Calculate confidence score for detection"""
        # Check how close to a rectangle the shape is
        # by comparing diagonals and opposite sides
        
        # Calculate side lengths
        sides = []
        for i in range(4):
            side = np.linalg.norm(corners[(i + 1) % 4] - corners[i])
            sides.append(side)
            
        # Calculate diagonals
        diag1 = np.linalg.norm(corners[2] - corners[0])
        diag2 = np.linalg.norm(corners[3] - corners[1])
        
        # Perfect rectangle: opposite sides equal, diagonals equal
        side_diff = abs(sides[0] - sides[2]) / max(sides[0], sides[2])
        side_diff += abs(sides[1] - sides[3]) / max(sides[1], sides[3])
        diag_diff = abs(diag1 - diag2) / max(diag1, diag2)
        
        # Calculate confidence (1.0 for perfect rectangle)
        confidence = 1.0 - (side_diff + diag_diff) / 3.0
        
        return max(0.0, min(1.0, confidence))
        
    def _is_stable_detection(self, screen: ScreenRegion, movement_threshold: float = 50) -> bool:
        """Check if detection is stable compared to previous frames"""
        if self._last_screen is None:
            self._stable_frames = 1
            return True
            
        # Compare corner positions
        movement = np.mean(np.linalg.norm(
            screen.corners - self._last_screen.corners,
            axis=1
        ))
        
        if movement < movement_threshold:
            self._stable_frames += 1
        else:
            self._stable_frames = 1
            
        return self._stable_frames >= self._stability_threshold
        
    def refine_detection(self, frame: np.ndarray, screen: ScreenRegion) -> ScreenRegion:
        """Refine screen detection using additional processing"""
        # Extract screen region
        if screen.transform_matrix is not None:
            warped = cv2.warpPerspective(
                frame,
                screen.transform_matrix,
                (screen.width, screen.height)
            )
            
            # Analyze warped image for better corner detection
            gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            
            # Use Harris corner detection for refinement
            corners = cv2.goodFeaturesToTrack(
                gray_warped,
                maxCorners=4,
                qualityLevel=0.01,
                minDistance=min(screen.width, screen.height) * 0.3
            )
            
            if corners is not None and len(corners) == 4:
                # Transform corners back to original image space
                corners = corners.reshape(-1, 2)
                corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
                
                # Inverse transform
                inv_transform = np.linalg.inv(screen.transform_matrix)
                original_corners = []
                
                for corner in corners_homogeneous:
                    transformed = inv_transform @ corner
                    original_corners.append(transformed[:2] / transformed[2])
                    
                refined_corners = np.array(original_corners)
                return self._create_screen_region(refined_corners, frame.shape)
                
        return screen
        
    def draw_detection(self, frame: np.ndarray, screen: ScreenRegion) -> np.ndarray:
        """Draw detection overlay on frame"""
        overlay = frame.copy()
        
        # Draw screen outline
        cv2.drawContours(overlay, [screen.corners.astype(int)], -1, (0, 255, 0), 3)
        
        # Draw corner points
        for i, corner in enumerate(screen.corners):
            cv2.circle(overlay, tuple(corner.astype(int)), 8, (0, 0, 255), -1)
            
        # Draw confidence
        cv2.putText(
            overlay,
            f"Screen: {screen.confidence:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return overlay