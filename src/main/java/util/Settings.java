package util;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;

import java.io.StringReader;
import java.util.Date;

/**
 * Created by hp on 7/7/2016.
 */
public class Settings {


    private String userName;
    private String password;
    private boolean activated;
    private long activationDate;


    public Settings(){}
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
     * @param settingsJson A json describing a valid {@link Settings} data
     * @return an {@link Settings} object that encapsulates the json.
     */
    public static Settings parseJson(String settingsJson){
        JsonReader reader = new JsonReader(new StringReader(settingsJson));
        return new Gson().fromJson(reader , Settings.class);
    }

    @Override
    public String toString() {
        return new Gson().toJson(this);
    }
}
