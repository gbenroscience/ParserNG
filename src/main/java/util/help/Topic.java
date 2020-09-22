package util.help;


import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;

import java.io.StringReader;

/**
 * Created by JIBOYE Oluwagbemiro Olaoluwa on 8/5/2016.
 */
public class Topic{

    private String title;
    private String content;


    public Topic(String title,String content){
        this.title = title;
        this.content = content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public String getContent() {
        return content;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getTitle() {
        return title;
    }

    public static Topic parseTopic(String json){
        JsonReader reader = new JsonReader(new StringReader(json));
        return new Gson().fromJson(reader, Topic.class);
    }

    @Override
    public String toString() {
        return new Gson().toJson(this);
    }
}
