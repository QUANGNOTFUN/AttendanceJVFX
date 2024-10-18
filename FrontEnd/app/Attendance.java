package app;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

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

public class Attendance {

	// Đường dẫn tới tệp cascade phân loại khuôn mặt
	private static final String FACE_CASCADE_PATH = "open_cv/haarcascade_frontalface_alt.xml";
	private static final String DATASET_PATH = "dataset"; // Đường dẫn tới thư mục dataset

	private CascadeClassifier faceCascade;
	private Map<Integer, String> studentMap; // Lưu trữ ID và tên sinh viên
	private Map<Integer, Mat> faceHistograms; // Lưu trữ histogram của khuôn mặt

	public Attendance() {
		// Khởi tạo bộ phân loại khuôn mặt
		faceCascade = new CascadeClassifier(FACE_CASCADE_PATH);
		studentMap = new HashMap<>();
		faceHistograms = new HashMap<>();
		loadStudentData(); // Tải dữ liệu sinh viên và tính histogram từ dataset
	}

	public static void main(String[] args) {
		// Nạp thư viện OpenCV
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.out.println("OpenCV library loaded.");

		Attendance faceRecognizer = new Attendance();
		faceRecognizer.startRecognition();
	}

	// Hàm tải dữ liệu sinh viên từ thư mục dataset
	private void loadStudentData() {
		File datasetDir = new File(DATASET_PATH);
		File[] studentDirs = datasetDir.listFiles(File::isDirectory); // Lấy danh sách thư mục sinh viên

		if (studentDirs != null) {
			for (File studentDir : studentDirs) {
				String studentName = studentDir.getName(); // Lấy tên thư mục
				String[] parts = studentName.split("_"); // Giả sử tên thư mục có định dạng id_name

				// Thêm kiểm tra tên thư mục có chứa ký tự đặc biệt
				if (!studentName.matches("^[a-zA-Z0-9_]+$")) {
					System.out.println("Tên thư mục chứa ký tự không hợp lệ: " + studentName);
					continue; // Bỏ qua thư mục không hợp lệ
				}

				int id;
				try {
					id = Integer.parseInt(parts[0]); // Lấy ID từ phần trước dấu gạch dưới
					studentMap.put(id, studentName); // Lưu ID và tên sinh viên
					System.out.println("Đã thêm sinh viên: ID = " + id + ", Tên = " + studentName);

					// Tải ảnh khuôn mặt từ thư mục sinh viên
					File[] faceImages = studentDir.listFiles((dir, name) -> name.endsWith(".jpg"));
					if (faceImages != null && faceImages.length > 0) {
						// Tính toán histogram cho ảnh đầu tiên của sinh viên
						Mat image = Imgcodecs.imread(faceImages[0].getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
						Mat histogram = new Mat();
						calculateHistogram(image, histogram);
						faceHistograms.put(id, histogram); // Lưu histogram cho sinh viên
					}

				} catch (NumberFormatException e) {
					System.out.println("ID không hợp lệ trong tên thư mục: " + studentName);
				}
			}
		}
	}

	// Hàm bắt đầu nhận diện khuôn mặt
	public void startRecognition() {
		VideoCapture camera = new VideoCapture(0);
		if (!camera.isOpened()) {
			System.out.println("Không thể mở camera.");
			return;
		}

		Mat frame = new Mat();
		while (true) {
			if (camera.read(frame)) {
				// Lật khung hình
				Core.flip(frame, frame, 1); // Lật khung hình theo chiều ngang

				// Phát hiện khuôn mặt
				detectAndDisplayFaces(frame);

				// Hiển thị khung hình lên cửa sổ
				HighGui.imshow("Camera", frame);

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

	// Hàm phát hiện khuôn mặt và vẽ khung xung quanh khuôn mặt
	private void detectAndDisplayFaces(Mat frame) {
		Mat grayFrame = new Mat();

		// Chuyển đổi khung hình sang ảnh xám
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(grayFrame, grayFrame); // Cải thiện chất lượng ảnh xám

		// Phát hiện khuôn mặt
		MatOfRect faces = new MatOfRect();
		faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0, new Size(30, 30), new Size());

		// Chuyển đổi MatOfRect thành mảng Rect[] để duyệt
		Rect[] facesArray = faces.toArray();

		// Vẽ khung xung quanh khuôn mặt và hiển thị tên
		for (Rect face : facesArray) {
			Imgproc.rectangle(frame, face.tl(), face.br(), new Scalar(0, 255, 0), 2);

			// Xác định sinh viên và giá trị khớp
			Map.Entry<Integer, Double> result = identifyStudent(face, frame);
			int studentId = result.getKey();
			double matchValue = result.getValue();

			if (studentId != -1) {
				String studentName = studentMap.get(studentId);

				// Tính toán phần trăm độ chính xác
				double accuracy = Math.max(0, 100 - (matchValue / 200.0 * 100)); // Từ 0 đến 100%
				String accuracyText = String.format("Accuracy: %.2f%%", accuracy);

				// Hiển thị tên sinh viên và độ chính xác
				Imgproc.putText(frame, studentName, new Point(face.x, face.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
						new Scalar(0, 255, 0), 2);
				Imgproc.putText(frame, accuracyText, new Point(face.x, face.y + face.height + 20),
						Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
				Imgproc.putText(frame, "ID: " + studentId, new Point(face.x, face.y + face.height + 40),
						Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
			} else {
				// Hiển thị "Unknown" nếu không có sinh viên nào khớp
				Imgproc.putText(frame, "Unknown", new Point(face.x, face.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
						new Scalar(0, 0, 255), 2);
			}
		}
	}

	// Hàm xác định ID sinh viên từ khuôn mặt
	private Map.Entry<Integer, Double> identifyStudent(Rect face, Mat frame) {
		// Cắt khuôn mặt từ khung hình
		Mat faceMat = new Mat(frame, face);
		Mat histogram = new Mat();
		calculateHistogram(faceMat, histogram); // Tính toán histogram cho khuôn mặt

		int bestMatchId = -1;
		double bestMatchValue = Double.MAX_VALUE;
		double threshold = 200.0; // Ngưỡng để xác định khớp

		// So sánh với các histogram đã lưu
		for (Map.Entry<Integer, Mat> entry : faceHistograms.entrySet()) {
			double matchValue = Imgproc.compareHist(histogram, entry.getValue(), Imgproc.CV_COMP_CHISQR);
			if (matchValue < bestMatchValue) {
				bestMatchValue = matchValue;
				bestMatchId = entry.getKey();
			}
		}

		// Nếu giá trị khớp lớn hơn ngưỡng, trả về -1 để thể hiện không có sinh viên nào
		// khớp
		if (bestMatchValue > threshold) {
			bestMatchId = -1;
		}

		// Trả về cả ID và giá trị khớp tốt nhất
		return Map.entry(bestMatchId, bestMatchValue);
	}

	// Hàm tính toán histogram
	private void calculateHistogram(Mat image, Mat histogram) {
		MatOfInt histSize = new MatOfInt(256);
		MatOfFloat histRange = new MatOfFloat(0f, 256f);
		MatOfInt channels = new MatOfInt(0);

		Imgproc.calcHist(Arrays.asList(image), channels, new Mat(), histogram, histSize, histRange);
		Core.normalize(histogram, histogram, 0, 1, Core.NORM_MINMAX);
	}
}
