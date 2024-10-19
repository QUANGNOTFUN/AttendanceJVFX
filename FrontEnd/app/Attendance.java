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
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class Attendance {
    
    private static final String CASCADE_PATH = "open_cv/haarcascade_frontalface_alt.xml";
    private static final String STUDENT_DATA_PATH = "dataset"; // Đường dẫn đến thư mục chứa dataset
    private CascadeClassifier faceClassifier;
    private Map<Integer, String> studentNames; // Lưu tên sinh viên theo ID
    private Map<Integer, Mat> faceHistograms; // Lưu histogram của các khuôn mặt

    // Thay đổi để theo dõi thời gian và số lần thành công
    private long startTime = 0; // Thời gian bắt đầu nhận diện
    private int successCount = 0; // Số lần nhận diện thành công
    private static final int SUCCESS_THRESHOLD = 60; // Ngưỡng thành công (60%)
    private static final long DURATION = 3000; // 3 giây

    public Attendance() {
        faceClassifier = new CascadeClassifier(CASCADE_PATH);
        studentNames = new HashMap<>();
        faceHistograms = new HashMap<>();
        loadStudentData(); // Tải dữ liệu sinh viên và tính toán histogram
    }

    // Phương thức khởi động quá trình nhận diện khuôn mặt
    public void startRecognition() {
        System.out.println("Bắt đầu nhận diện khuôn mặt...");

        // Khởi tạo camera để lấy hình ảnh
        VideoCapture camera = new VideoCapture(0); // Sử dụng camera mặc định
        if (!camera.isOpened()) {
            System.out.println("Không thể mở camera.");
            return;
        }

        Mat frame = new Mat();
        while (true) {
            if (camera.read(frame)) {
                Core.flip(frame, frame, 1); // Lật khung hình để dễ dàng nhận diện

                // Phát hiện khuôn mặt trong ảnh
                detectAndDisplay(frame); // Phát hiện khuôn mặt và hiển thị

                // Hiển thị khung hình nhận diện
                HighGui.imshow("Nhận diện khuôn mặt", frame);

                // Nếu nhấn phím 'q' thì thoát
                if (HighGui.waitKey(1) == 'q') {
                    break;
                }
            } else {
                System.out.println("Không thể đọc khung hình.");
                break;
            }
        }

        // Giải phóng tài nguyên
        camera.release();
        HighGui.destroyAllWindows();
    }

    // Hàm phát hiện khuôn mặt và xử lý điểm danh
    private void detectAndDisplay(Mat frame) {
        Mat grayFrame = new Mat();
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(grayFrame, grayFrame); // Cải thiện chất lượng ảnh

        // Phát hiện khuôn mặt trong ảnh
        MatOfRect faces = new MatOfRect();
        faceClassifier.detectMultiScale(grayFrame, faces, 1.1, 3, 0, new Size(100, 100), new Size());

        Rect[] faceArray = faces.toArray();
        for (Rect face : faceArray) {
            Imgproc.rectangle(frame, face.tl(), face.br(), new Scalar(0, 255, 0), 2); // Vẽ khung quanh khuôn mặt

            // Nhận diện và xác định sinh viên
            Map.Entry<Integer, Double> result = recognizeFace(face, frame);
            int studentId = result.getKey();
            double matchValue = result.getValue();

            if (studentId != -1) {
                String studentName = studentNames.get(studentId);
                String accuracyText = String.format("Accuracy: %.0f%%", (100 - matchValue / 200.0 * 100));

                Imgproc.putText(frame, studentName, new Point(face.x, face.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
                Imgproc.putText(frame, accuracyText, new Point(face.x, face.y + face.height + 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);

                // Cập nhật điểm danh nếu độ chính xác trên 60%
                if (matchValue < 200) {
                    updateAttendance(studentId);
                }

                // Theo dõi độ chính xác để đảm bảo đạt yêu cầu 3 giây liên tục
                trackRecognition(studentId, matchValue);
            } else {
                Imgproc.putText(frame, "Unknown", new Point(face.x, face.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 255), 2);
                resetRecognition(); // Reset nếu không nhận diện được
            }
        }
    }

    // Hàm theo dõi độ chính xác của nhận diện
    private void trackRecognition(int studentId, double matchValue) {
        if (matchValue < 200) { // Nếu độ chính xác đủ
            if (startTime == 0) {
                startTime = System.currentTimeMillis(); // Ghi lại thời gian bắt đầu
            }

            // Kiểm tra xem thời gian đã đủ 3 giây chưa
            if (System.currentTimeMillis() - startTime >= DURATION) {
                // Điểm danh nếu đạt yêu cầu
                updateAttendance(studentId);
                resetRecognition(); // Reset lại
            }
        } else {
            resetRecognition(); // Reset nếu độ chính xác không đủ
        }
    }

    // Reset trạng thái nhận diện
    private void resetRecognition() {
        startTime = 0;
        successCount = 0;
    }

    // Hàm nhận diện và trả về ID sinh viên cùng độ chính xác
    private Map.Entry<Integer, Double> recognizeFace(Rect face, Mat frame) {
        Mat faceMat = new Mat(frame, face);
        Mat faceHistogram = new Mat();
        computeHistogram(faceMat, faceHistogram);

        int bestMatchId = -1;
        double bestMatchValue = Double.MAX_VALUE;

        // So sánh với tất cả các histogram của sinh viên
        for (Map.Entry<Integer, Mat> entry : faceHistograms.entrySet()) {
            double matchValue = Imgproc.compareHist(faceHistogram, entry.getValue(), Imgproc.CV_COMP_CHISQR);
            if (matchValue < bestMatchValue) {
                bestMatchValue = matchValue;
                bestMatchId = entry.getKey();
            }
        }

        // Nếu độ chính xác thấp, coi như không nhận diện được
        if (bestMatchValue > 200.0) {
            bestMatchId = -1;
        }

        return Map.entry(bestMatchId, bestMatchValue);
    }

    // Hàm tính toán histogram của khuôn mặt
    private void computeHistogram(Mat image, Mat histogram) {
        MatOfInt histSize = new MatOfInt(256);
        MatOfFloat histRange = new MatOfFloat(0f, 256f);
        MatOfInt channels = new MatOfInt(0);

        Imgproc.calcHist(Arrays.asList(image), channels, new Mat(), histogram, histSize, histRange);
        Core.normalize(histogram, histogram, 0, 1, Core.NORM_MINMAX);
    }

    // Cập nhật trạng thái điểm danh trong cơ sở dữ liệu
    private void updateAttendance(int studentId) {
        System.out.println("Sinh viên " + studentId + " đã được điểm danh.");
        // Gọi phương thức để cập nhật cơ sở dữ liệu tại đây
    }

    // Tải dữ liệu sinh viên từ thư mục dataset
    private void loadStudentData() {
        File datasetDir = new File(STUDENT_DATA_PATH);
        File[] studentDirs = datasetDir.listFiles(File::isDirectory); // Lấy tất cả thư mục sinh viên

        if (studentDirs != null) {
            for (File studentDir : studentDirs) {
                String studentName = studentDir.getName();
                String[] parts = studentName.split("_");
                try {
                    int studentId = Integer.parseInt(parts[0]);
                    studentNames.put(studentId, studentName);

                    // Tính histogram cho các ảnh của sinh viên
                    File[] images = studentDir.listFiles((dir, name) -> name.endsWith(".jpg"));
                    if (images != null) {
                        Mat totalHistogram = new Mat();
                        for (File image : images) {
                            Mat img = Imgcodecs.imread(image.getAbsolutePath());
                            computeHistogram(img, totalHistogram);
                        }
                        faceHistograms.put(studentId, totalHistogram);
                    }
                } catch (NumberFormatException e) {
                    System.out.println("Tên thư mục không hợp lệ: " + studentName);
                }
            }
        }
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Attendance attendance = new Attendance();
        attendance.startRecognition();
    }
}
