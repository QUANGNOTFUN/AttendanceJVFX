package controller;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class FaceAttendance {

	private static final String FACE_CASCADE_PATH = "open_cv/haarcascade_frontalface_alt.xml";
	private static final String DATASET_PATH = "dataset"; // Đường dẫn tới thư mục dataset
	private static final int RECOGNITION_THRESHOLD = 70; // Ngưỡng nhận diện
	private static final int RECOGNITION_TIME = 5000; // Thời gian nhận diện liên tục (5 giây)

	private CascadeClassifier faceCascade;
	private Map<Integer, String> studentMap;
	private Map<Integer, Mat> faceHistograms;

	public FaceAttendance() {
		// Khởi tạo bộ phân loại khuôn mặt
		faceCascade = new CascadeClassifier(FACE_CASCADE_PATH);
		studentMap = new HashMap<>();
		faceHistograms = new HashMap<>();
		loadStudentData(); // Tải dữ liệu sinh viên
	}

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.out.println("OpenCV library loaded.");

		FaceAttendance faceAttendance = new FaceAttendance();
		faceAttendance.startRecognition();
	}

	private void loadStudentData() {
		File datasetDir = new File(DATASET_PATH);
		File[] studentDirs = datasetDir.listFiles(File::isDirectory); // Lấy danh sách thư mục sinh viên

		if (studentDirs != null) {
			for (File studentDir : studentDirs) {
				String studentName = studentDir.getName();
				String[] parts = studentName.split("_");

				if (parts.length < 2) {
					System.out.println("Tên thư mục không hợp lệ: " + studentName);
					continue;
				}

				int id;
				try {
					id = Integer.parseInt(parts[0]);
					studentMap.put(id, studentName);

					// Tải ảnh khuôn mặt từ thư mục sinh viên
					File[] faceImages = studentDir.listFiles((dir, name) -> name.endsWith(".jpg"));
					if (faceImages != null && faceImages.length > 0) {
						Mat image = Imgcodecs.imread(faceImages[0].getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
						Mat histogram = new Mat();
						calculateHistogram(image, histogram);
						faceHistograms.put(id, histogram);
					}
				} catch (NumberFormatException e) {
					System.out.println("ID không hợp lệ trong tên thư mục: " + studentName);
				}
			}
		}
	}

	public void startRecognition() {
		VideoCapture camera = new VideoCapture(0);
		if (!camera.isOpened()) {
			System.out.println("Không thể mở camera.");
			return;
		}

		Mat frame = new Mat();
		AtomicInteger detectedCount = new AtomicInteger(0);
		long startTime = 0;

		while (true) {
			if (camera.read(frame)) {
				Core.flip(frame, frame, 1); // Lật khung hình theo chiều ngang
				detectAndDisplayFaces(frame, detectedCount); // Phát hiện khuôn mặt

				// Hiển thị khung hình lên cửa sổ
				HighGui.imshow("Camera", frame);

				// Kiểm tra tỉ lệ nhận diện
				if (detectedCount.get() > 0) {
					if (startTime == 0) {
						startTime = System.currentTimeMillis(); // Bắt đầu đếm thời gian
					} else if (System.currentTimeMillis() - startTime >= RECOGNITION_TIME) {
						System.out.println("Điểm danh thành công!");
						// Ghi nhận điểm danh vào cơ sở dữ liệu (cần thực hiện thêm)
						detectedCount.set(0); // Đặt lại số lượng phát hiện
						startTime = 0; // Đặt lại thời gian
					}
				} else {
					startTime = 0; // Đặt lại thời gian nếu không có khuôn mặt phát hiện
				}

				// Thoát nếu nhấn phím 'q'
				if (HighGui.waitKey(1) == 'q') {
					break;
				}
			} else {
				System.out.println("Không thể đọc khung hình từ camera.");
				break;
			}
		}

		camera.release();
		HighGui.destroyAllWindows();
	}

	public void detectAndDisplayFaces(Mat frame, AtomicInteger detectedCount) {
		Mat grayFrame = new Mat();
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(grayFrame, grayFrame);

		MatOfRect faces = new MatOfRect();
		faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0, new Size(30, 30), new Size());

		Rect[] facesArray = faces.toArray();

		for (Rect face : facesArray) {
			Imgproc.rectangle(frame, face.tl(), face.br(), new Scalar(0, 255, 0), 2);
			Map.Entry<Integer, Double> result = identifyStudent(face, frame);
			int studentId = result.getKey();
			double matchValue = result.getValue();

			if (studentId != -1) {
				detectedCount.incrementAndGet(); // Tăng số lượng phát hiện
				String studentName = studentMap.get(studentId);
				double accuracy = Math.max(0, 100 - (matchValue / 200.0 * 100));
				Imgproc.putText(frame, studentName, new Point(face.x, face.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
						new Scalar(0, 255, 0), 2);
				Imgproc.putText(frame, String.format("Accuracy: %.2f%%", accuracy),
						new Point(face.x, face.y + face.height + 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
						new Scalar(0, 255, 0), 2);
				Imgproc.putText(frame, "ID: " + studentId, new Point(face.x, face.y + face.height + 40),
						Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
			} else {
				Imgproc.putText(frame, "Unknown", new Point(face.x, face.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
						new Scalar(0, 0, 255), 2);
			}
		}
	}

	private Map.Entry<Integer, Double> identifyStudent(Rect face, Mat frame) {
		Mat faceMat = new Mat(frame, face);
		Mat histogram = new Mat();
		calculateHistogram(faceMat, histogram);

		int bestMatchId = -1;
		double bestMatchValue = Double.MAX_VALUE;
		double threshold = 200.0; // Ngưỡng để xác định khớp

		for (Map.Entry<Integer, Mat> entry : faceHistograms.entrySet()) {
			double matchValue = Imgproc.compareHist(histogram, entry.getValue(), Imgproc.CV_COMP_CHISQR);
			if (matchValue < bestMatchValue) {
				bestMatchValue = matchValue;
				bestMatchId = entry.getKey();
			}
		}

		if (bestMatchValue > threshold) {
			bestMatchId = -1;
		}

		return Map.entry(bestMatchId, bestMatchValue);
	}

	private void calculateHistogram(Mat image, Mat histogram) {
		MatOfInt histSize = new MatOfInt(256);
		MatOfFloat histRange = new MatOfFloat(0f, 256f);
		MatOfInt channels = new MatOfInt(0);

		Imgproc.calcHist(Arrays.asList(image), channels, new Mat(), histogram, histSize, histRange);
		Core.normalize(histogram, histogram, 0, 1, Core.NORM_MINMAX);
	}
}
