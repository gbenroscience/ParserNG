package util.help;




import interfaces.Savable;
import java.io.StringReader;
import util.Serializer;

/**
 * Created by JIBOYE Oluwagbemiro Olaoluwa on 8/5/2016.
 */
public class Topic implements Savable{

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

    public static Topic parseTopic(String enc){
        return (Topic) Serializer.deserialize(enc);
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
