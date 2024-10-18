package connect;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class ConnectSQL {
	private Connection connection;

	// Phương thức để tạo kết nối với database
	public boolean connect_db() {
		String url = "jdbc:mysql://localhost:3306/face_id";
		String username = "root";
		String password = "";

		try {
			connection = DriverManager.getConnection(url, username, password);
			System.out.println("Kết nối database thành công!");
			return true;
		} catch (SQLException e) {
			System.err.println("Kết nối database thất bại!");
			e.printStackTrace();
			return false;
		}
	}

	public Connection getConnection() {
		return connection;
	}

	// Phương thức để thêm sinh viên vào cơ sở dữ liệu
	public void addStudentToDatabase(int id, String name, String studentClass, String faceImgPath) {
		String query = "INSERT INTO students (id, name, studentClass, faceImgPath) VALUES (?, ?, ?, ?)";

		try (PreparedStatement preparedStatement = connection.prepareStatement(query)) {
			preparedStatement.setInt(1, id);
			preparedStatement.setString(2, name);
			preparedStatement.setString(3, studentClass);
			preparedStatement.setString(4, faceImgPath);
			preparedStatement.executeUpdate();
			System.out.println("Đã lưu thông tin sinh viên vào cơ sở dữ liệu.");
		} catch (SQLException e) {
			System.err.println("Lỗi khi lưu thông tin sinh viên: " + e.getMessage());
		}
	}

	public int getNextStudentId() {
		int nextId = 1; // Giá trị mặc định
		String query = "SELECT MAX(id) AS max_id FROM students"; // Giả sử bạn có bảng students

		try (Statement stmt = connection.createStatement(); ResultSet rs = stmt.executeQuery(query)) {
			if (rs.next()) {
				nextId = rs.getInt("max_id") + 1; // Tăng ID lên 1
			}
		} catch (SQLException e) {
			System.err.println("Có lỗi xảy ra khi lấy ID tiếp theo: " + e.getMessage());
		}
		return nextId; // Trả về ID tiếp theo
	}

	// Phương thức để cập nhật trạng thái sinh viên
	public void updateStudentStatus(int id, String status) {
		String query = "UPDATE students SET status = ? WHERE id = ?";

		try (PreparedStatement preparedStatement = connection.prepareStatement(query)) {
			preparedStatement.setString(1, status);
			preparedStatement.setInt(2, id);
			preparedStatement.executeUpdate();
			System.out.println("Đã cập nhật trạng thái sinh viên thành công.");
		} catch (SQLException e) {
			System.err.println("Lỗi khi cập nhật trạng thái sinh viên: " + e.getMessage());
		}
	}

}
