package util;


import interfaces.Savable;

/**
 * Created by hp on 7/7/2016.
 */
public class Settings implements Savable {

    private String userName;
    private String password;
    private boolean activated;
    private long activationDate;

    public Settings() {
    }

    public Settings(String password, boolean activated, String userName, long activationDate) {
        this.password = password;
        this.activated = activated;
        this.userName = userName;
        this.activationDate = activationDate;
    }

    public String getUserName() {
        return userName;
    }

    public void setUserName(String userName) {
        this.userName = userName;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public boolean isActivated() {
        return activated;
    }

    public void setActivated(boolean activated) {
        this.activated = activated;
    }

    public long getActivationDate() {
        return activationDate;
    }

    public void setActivationDate(long activationDate) {
        this.activationDate = activationDate;
    }

    /**
     *
     * @param enc The encoded format of the byte array: [num1, num2, num3, num4,
     * ...]
     * @return the Variable object that represents the encoded data
     */
    public static Settings parse(String enc) {
        return (Settings) Serializer.deserialize(enc);
    }

    @Override
    public String serialize() {
           return Serializer.serialize(this);
    }
    
    
  

    @Override
    public String toString() {
        return Serializer.serialize(this);
    }
}
