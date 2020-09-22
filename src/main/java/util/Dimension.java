package util;

public class Dimension {

	public int width;
	
	public int height;
	public Dimension() {
	 
	}
	public Dimension(int width,int height){
		this.width = width;
		this.height = height;
	}
	public Dimension(Dimension d){
		this.width = d.width;
		this.height = d.height;
	}
	
	public void setWidth(int width) {
		this.width = width;
	}
	public int getWidth() {
		return width;
	}
	public void setHeight(int height) {
		this.height = height;
	}
	public int getHeight() {
		return height;
	}
	
	
}
