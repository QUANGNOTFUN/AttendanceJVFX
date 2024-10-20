package app;

import org.opencv.core.Core;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public class MainApp extends Application {

    private boolean isExited = false; // Cờ để theo dõi trạng thái thoát

    @Override
    public void start(Stage primaryStage) {
        // Tạo các nút
        Button diemDanhButton = new Button("Điểm danh");
        Button adminButton = new Button("Admin");
        Button thoatButton = new Button("Thoát");

        // Đặt hành động cho nút "Điểm danh" để mở cửa sổ Attendance
        diemDanhButton.setOnAction(e -> {
            Attendance attendance = new Attendance();
            Stage attendanceStage = new Stage(); // Tạo cửa sổ mới
            attendance.startRecognition(); // Mở Attendance trong cửa sổ mới

            // Đặt tiêu đề và kích thước cho cửa sổ Attendance
            attendanceStage.setTitle("Điểm danh");
            attendanceStage.setWidth(800); // Thiết lập chiều rộng
            attendanceStage.setHeight(600); // Thiết lập chiều cao
            attendanceStage.show(); // Hiện thị cửa sổ mới
        });

        // Đặt hành động cho nút "Admin" để mở cửa sổ Admin
        adminButton.setOnAction(e -> {
            Admin admin = new Admin();
            Stage adminStage = new Stage(); // Tạo cửa sổ mới cho Admin
            admin.start(adminStage); // Mở Admin trong cửa sổ mới

            // Đặt tiêu đề cho cửa sổ Admin
            adminStage.setTitle("Admin");
            adminStage.show(); // Hiện thị cửa sổ mới
        });

        // Đặt hành động cho nút "Thoát" để thoát ứng dụng
        thoatButton.setOnAction(e -> {
            if (!isExited) { // Kiểm tra xem thông báo đã hiển thị chưa
                System.out.println("Thoát ứng dụng"); // In thông báo khi thoát
                isExited = true; // Đánh dấu rằng thông báo đã hiển thị
            }
            primaryStage.close(); // Đóng cửa sổ ứng dụng chính
        });

        // Tạo VBox để chứa các nút
        VBox layout = new VBox(20); // Khoảng cách giữa các nút là 10
        layout.getChildren().addAll(diemDanhButton, adminButton, thoatButton);

        // Đặt style cho layout
        layout.setStyle("-fx-padding: 20; -fx-alignment: center;");

        // Tạo scene và thiết lập cho stage
        Scene scene = new Scene(layout, 300, 200);
        scene.getStylesheets().add(getClass().getResource("style.css").toExternalForm()); // Liên kết đến CSS
        primaryStage.setScene(scene);
        primaryStage.setTitle("Ứng dụng Điểm danh");
        primaryStage.show();
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("OpenCV library loaded.");
        launch(args);
    }
}
