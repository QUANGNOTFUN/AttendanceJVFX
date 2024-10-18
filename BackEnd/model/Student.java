package model;

public class Student {
	private int id; // id sinh viên
	private String name; // tên sinh viên
	private String studentClass; // lớp sinh viên
	private String faceImgPath; // đường dẫn ảnh khuôn mặt
	private String status; // Trạng thái sinh viên

	public Student(int id, String name, String studentClass, String faceImgPath) {
		this.id = id;
		this.name = name;
		this.studentClass = studentClass;
		this.faceImgPath = faceImgPath;
		this.status = "absent"; // Mặc định là absent
	}

	// Getter và Setter
	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public String getName() {
		return name;
	}

	public String getStudentClass() {
		return studentClass;
	}

	public String getFaceImgPath() {
		return faceImgPath;
	}

	public void setFaceImgPath(String faceImgPath) {
		this.faceImgPath = faceImgPath;
	}

	public String getStatus() {
		return status;
	}

	public void setStatus(String status) {
		this.status = status;
	}

	public void displayStudentInfo() {
		System.out.println("ID: " + id);
		System.out.println("Tên: " + name);
		System.out.println("Lớp: " + studentClass);
		System.out.println("Đường dẫn ảnh: " + faceImgPath);
	}
}
