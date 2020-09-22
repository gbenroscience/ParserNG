package util;

public class SimplePoint {

	public int x;
	public int y;
	
	public SimplePoint() {

     }
	
	public SimplePoint(int x, int y){
		this.x = x;
		this.y = y;
	}
	public SimplePoint(SimplePoint p){
		this.x = p.x;
		this.y = p.y;
	}
	
	
	
	
	
	public void setX(int x) {
		this.x = x;
	}
	public int getX() {
		return x;
	}
	
	public void setY(int y) {
		this.y = y;
	}
	public int getY() {
		return y;
	}
}
