package app;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.opencv.core.Core;
import javafx.scene.control.ComboBox;

import connect.ConnectSQL;
import controller.FaceDetect;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.stage.Stage;
import model.Student;

public class AddStudent extends Application {

    private TextField nameField;
    private ComboBox<String> classComboBox; // Thay đổi từ TextField sang ComboBox
    private FaceDetect faceDetect; 
    private ConnectSQL connectSQL;

    @Override
    public void start(Stage primaryStage) {
        // Tạo trường nhập liệu cho tên
        nameField = new TextField();
        
        // Tạo ComboBox cho lớp
        classComboBox = new ComboBox<>();
        classComboBox.getItems().addAll("CN22A", "CN22B", "CN22C", "CN22D"); // Thêm lớp vào ComboBox

        // Tạo nút thêm sinh viên
        Button addButton = new Button("Thêm sinh viên");
        addButton.setOnAction(e -> addStudent());

        // Tạo giao diện
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.add(new Label("Tên sinh viên:"), 0, 0);
        grid.add(nameField, 1, 0);
        grid.add(new Label("Lớp sinh viên:"), 0, 1);
        grid.add(classComboBox, 1, 1); // Thay đổi từ classField sang classComboBox
        grid.add(addButton, 1, 2);

        Scene scene = new Scene(grid, 300, 200);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Thêm Sinh Viên");
        primaryStage.show();
    }

    private void addStudent() {
        try {
            // Nhập thông tin sinh viên từ các trường
            String name = nameField.getText();
            String studentClass = classComboBox.getValue(); // Lấy giá trị từ ComboBox

            if (studentClass == null) {
                System.out.println("Vui lòng chọn lớp cho sinh viên.");
                return; // Nếu không chọn lớp, dừng lại
            }

            // Kết nối đến cơ sở dữ liệu để lấy ID tiếp theo
            connectSQL = new ConnectSQL();
            if (connectSQL.connect_db()) {
                // Lấy ID của lớp dựa trên tên lớp
                int classID = connectSQL.getClassIdByName(studentClass);
                if (classID == -1) {
                    System.out.println("Không tìm thấy ID của lớp.");
                    return;
                }

                // Tạo đối tượng Student với ID tạm thời
                Student student = new Student(0, name, studentClass, null); 

                // Lấy ID mới từ cơ sở dữ liệu
                int newId = connectSQL.getNextStudentId(); 
                student.setId(newId); 

                // Tạo thư mục để lưu ảnh khuôn mặt
                String studentDirectory = "dataset/" + newId; 
                File directory = new File(studentDirectory);
                if (!directory.exists()) {
                    directory.mkdirs(); 
                }

                // Khởi tạo và chạy FaceDetect để chụp ảnh
                faceDetect = new FaceDetect();
                faceDetect.detectFace(student);

                // Lấy đường dẫn ảnh đã lưu trong đối tượng student
                String imagePath = student.getFaceImgPath();
                if (imagePath != null) {
                    // Lưu thông tin sinh viên vào cơ sở dữ liệu
                    connectSQL.addStudentToDatabase(newId, name, studentClass, imagePath);

                    // Định nghĩa currentDate
                    String currentDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());

                    // Lưu thông tin điểm danh mà không cần truyền `status`
                    connectSQL.addAttendanceToDatabase(newId, currentDate, currentDate, connectSQL.getClassIdByName(studentClass));

                    // Hiển thị thông tin sinh viên
                    student.displayStudentInfo();
                }

                // Làm sạch các trường nhập liệu sau khi thêm
                nameField.clear();
                classComboBox.setValue(null); // Đặt lại giá trị ComboBox
            }
        } catch (Exception e) {
            System.out.println("Có lỗi xảy ra: " + e.getMessage());
        }
    }


    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        launch(args);
    }
}
