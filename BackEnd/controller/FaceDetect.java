package controller;

import java.io.File;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import model.Student;

public class FaceDetect {
    private VideoCapture camera;
    private CascadeClassifier faceDetector;
    private String datasetDir = "dataset";
    private int faceCount = 1;
    private double recognitionAccuracy; // Field to store recognition accuracy

    // Constructor
    public FaceDetect() {
        // Nạp thư viện OpenCV
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Khởi tạo camera
        this.camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("Không thể mở camera.");
            return;
        }

        // Khởi tạo bộ phát hiện khuôn mặt
        this.faceDetector = new CascadeClassifier("open_cv/haarcascade_frontalface_default.xml");

        // Tạo thư mục dataset nếu chưa có
        File directory = new File(datasetDir);
        if (!directory.exists()) {
            directory.mkdir();
        }

        this.recognitionAccuracy = 0.0; // Initialize accuracy
    }

    // Phương thức tạo thư mục cho sinh viên
    private void createStudentDirectory(int studentId, String studentName) {
        String studentDir = datasetDir + "/" + studentId + "_" + studentName; // Use id_name format
        File directory = new File(studentDir);
        if (!directory.exists()) {
            directory.mkdir();
        }
    }

    // Phương thức phát hiện và lưu khuôn mặt
    public void detectFace(Student student) { // Accept Student object
        Mat frame = new Mat();

        // Tạo thư mục cho sinh viên
        createStudentDirectory(student.getId(), student.getName());

        String studentDir = datasetDir + "/" + student.getId() + "_" + student.getName();

        try {
            while (true) {
                if (camera.read(frame)) {
                    Core.flip(frame, frame, 1); // Lật ảnh theo chiều ngang

                    // Chuyển đổi ảnh sang màu xám
                    Mat grayFrame = new Mat();
                    Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

                    MatOfRect facesArray = new MatOfRect();
                    faceDetector.detectMultiScale(grayFrame, facesArray);

                    // Lưu khuôn mặt chỉ khi khuôn mặt đủ lớn
                    Rect[] faces = facesArray.toArray();
                    for (Rect face : faces) {
                        // Kiểm tra kích thước khuôn mặt
                        if (face.width >= 250 && face.height >= 250) {
                            // Vẽ khung bao quanh khuôn mặt
                            Imgproc.rectangle(frame, new Point(face.x, face.y),
                                    new Point(face.x + face.width, face.y + face.height), new Scalar(0, 255, 0), 2);

                            // Lưu khuôn mặt vào file dưới dạng ảnh xám
                            Mat faceImage = new Mat(grayFrame, face); // Lấy khuôn mặt từ ảnh xám thay vì ảnh màu
                            String filename = studentDir + "/face_" + faceCount + ".jpg";
                            try {
                                Imgcodecs.imwrite(filename, faceImage);
                                System.out.println("Đã lưu khuôn mặt vào: " + filename);
                                student.setFaceImgPath(filename); // Lưu đường dẫn ảnh vào đối tượng student

                                // Hiển thị thông báo ảnh đã chụp lên cửa sổ camera
                                Imgproc.putText(frame, "Captured: " + filename, new Point(10, 30),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);
                            } catch (Exception e) {
                                System.out.println("Không thể lưu khuôn mặt: " + e.getMessage());
                            }

                            // Tăng số lượng ảnh đã lưu
                            faceCount++;

                            // Dừng sau khi đã chụp được một khuôn mặt
                            break; // Dừng vòng lặp sau khi đã chụp một khuôn mặt đủ lớn
                        }
                    }

                    // Hiển thị khung hình camera sau khi chụp ảnh
                    HighGui.imshow("Face Detection", frame);
                    if (HighGui.waitKey(1) == 'q' || faceCount > 10) {
                        break; // Dừng nếu nhấn 'q' hoặc đã chụp đủ 10 khuôn mặt
                    }

                    // Dừng 0.4 giây sau mỗi lần chụp
                    try {
                        Thread.sleep(400); // 400 milliseconds
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                } else {
                    System.out.println("Không thể đọc khung hình.");
                    break;
                }
            }
        } catch (Exception e) {
            System.out.println("Có lỗi xảy ra trong quá trình phát hiện khuôn mặt: " + e.getMessage());
        } finally {
            camera.release();
            HighGui.destroyAllWindows();	
        }
    }

    // Method to get the current recognition accuracy
    public double getRecognitionAccuracy() {
        return recognitionAccuracy;
    }

    // Example method to calculate accuracy (you need to implement this)
    private double calculateAccuracy(Mat detectedFace, Student student) {
        // Implement your logic to calculate accuracy based on detected face
        // For example, you could compare the detected face with the student's face
        // image
        // Return a value between 0 and 100
        // Placeholder logic for demonstration
        return Math.random() * 100; // Random accuracy for demonstration purposes
    }
}
