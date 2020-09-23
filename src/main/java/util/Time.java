package util;

import parser.STRING;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.InputMismatchException;

/**
 * Objects of this class supply methods that may be used for telling time and
 * parsing String representations of time in dd:hh:mm:ss or hh:mm:ss or hh:mm formats
 * and translating them into seconds formats.The reverse is also possible:i.e translating time in seconds into dd:hh:mm:ss format.
 * When combined with  Timer objects,objects of this class can help in creating timing utilities for software.
 * @author GBENRO JIBOYE
 */
public class Time {
  private long days;
  private int hours;
  private int minutes;
  private int seconds;
  /**
   * No-argument constructor for creating objects of class Time.sets all the numerical properties of the object to 0
   */
  public Time(){
      this("00:00:00:00");
  }
  /**
   * This constructor creates objects of class Time and determines the values of various properties of the created based on the input time.
   * @param time the time property of the created object
   */
public Time(String time){
    //first get values from the string and fix in attributes.
format$ValidateTime(time);
//convert the values in the attributes into raw seconds
//in case the format is not proper e.g 00:32:87:128..is not proper formatting.
long secs = 86400*days + 3600*hours + 60*minutes + seconds;
//now convert the seconds format back to the proper string..and
//re-initialize the attributes.
format$ValidateTime(convertSecondsToTime(secs));
}//end constructor
/**
 *
 * @param days The days attribute.
 * @param hours The hours attribute.
 * @param minutes The minutes attribute.
 * @param seconds The seconds attribute.
 */
    public Time(long days, int hours, int minutes, int seconds) {
        long secs = 86400*days + 3600*hours + 60*minutes + seconds;
format$ValidateTime(convertSecondsToTime(secs));
    }
/**
 *
 * @param hours The hours attribute.
 * @param minutes The minutes attribute.
 * @param seconds The seconds attribute.
 */
    public Time( int hours, int minutes, int seconds) {
        long secs = 3600*hours + 60*minutes + seconds;
format$ValidateTime(convertSecondsToTime(secs));
    }

/**
 * @param minutes The minutes attribute.
 * @param seconds The seconds attribute.
 */
    public Time(int minutes, int seconds) {
        long secs =  60*minutes + seconds;
format$ValidateTime(convertSecondsToTime(secs));
    }

/**
 * @param seconds The seconds attribute.
 */
    public Time( int seconds) {
format$ValidateTime(convertSecondsToTime(seconds));
    }

  /**
   * This method determines the values associated with various properties of objects of this class dependent on the input time.
   * It accomplishes this by parsing and validating the input time, and from there assigns values to various properties of the
   * Time object based on the results of the parsing and validating.
   * The time is entered in the format "dd:hh:mm:ss" or "hh:mm:ss" or hh:mm
   * @param time the time property of the created object
   */
public void format$ValidateTime(String time){
            ArrayList<String> scanned=STRING.split(time, ":");
for(int i = 0; i < scanned.size(); i++){
scanned.set(i,scanned.get(i).trim());
}
//dd:hh:mm:ss
        if(scanned.size()==4){
            try{
            days=Integer.valueOf(scanned.get(0));
            hours=Integer.valueOf(scanned.get(1));
            minutes=Integer.valueOf(scanned.get(2));
            seconds=Integer.valueOf(scanned.get(3));
}
catch(NumberFormatException numErr){
throw new InputMismatchException("BAD TIME FORMAT!");
}
 }//end if
        //hh:mm:ss
        else if(scanned.size()==3){
            try{
            hours=Integer.valueOf(scanned.get(0));
            minutes=Integer.valueOf(scanned.get(1));
            seconds=Integer.valueOf(scanned.get(2));
            }
catch(NumberFormatException numErr){
throw new InputMismatchException("BAD TIME FORMAT!");
}
        }//end else if

                //hh:mm
        else if(scanned.size()==2){
            try{
            days=0;
            hours=Integer.valueOf(scanned.get(0));
            minutes=Integer.valueOf(scanned.get(1));
            }
catch(NumberFormatException numErr){
throw new InputMismatchException("BAD TIME FORMAT!");
}
        }//end else if

            String day=String.valueOf(days);
            String hour=String.valueOf(hours);
            String minute=String.valueOf(minutes);
            String second=String.valueOf(seconds);


            if(day.length()<2){
                day="0"+day;
            }
            if(hour.length()<2){
                hour="0"+hour;
            }
            if(minute.length()<2){
                minute="0"+minute;
            }
            if(second.length()<2){
                second="0"+second;
            }
            Long l = timeToSeconds();
}
                    //end method format$ValidateTime

/**
 *
 * @param t1 The Time object to combine subtractively
 * with this one.
 * @return a Time object that represents the absolute difference
 * between both Time objects.
 */
    public Time timeDiff( Time t1){
     Long  t_diff = Math.abs( timeToSeconds() - t1.timeToSeconds() );
     return new Time(convertSecondsToTime(t_diff));
    }

/**
 *
 * @param t1 The Time object to combine additively
 * with this one.
 * @return a time object that represents the absolute difference
 * between both Time objects.
 */
    public Time timeAddition( Time t1 ){
     Long  t_Add = Math.abs( timeToSeconds() + t1.timeToSeconds() );
     return new Time(convertSecondsToTime(t_Add));
    }
/**
 *
 * @return the current value of the time.
 * The returned time is not the current system time.To get the current system time,
 * make a call to method timeGetter()
 */
    public String getTime() {
        return days+":"+hours+":"+minutes+":"+seconds;
    }
/**
 *
 * @param time the value that we wish to change the time to manipulate to.
 */
    public void setTime(String time) {
        format$ValidateTime(time);
    }

    public long getDays() {
        return days;
    }

    public void setDays(long days) {
        this.days = days;
    }

    public int getHours() {
        return hours;
    }

    public void setHours(int hours) {
        this.hours = hours;
    }

    public int getMinutes() {
        return minutes;
    }

    public void setMinutes(int minutes) {
        this.minutes = minutes;
    }

    public int getSeconds() {
        return seconds;
    }

    public void setSeconds(int seconds) {
        this.seconds = seconds;
    }


/**
 *
 * @param t2 The other Time object to compare with this one.
 * @return true if this Time object is ahead of the other one.
 */
public boolean isAhead( Time t2 ){
    return this.timeToSeconds() > t2.timeToSeconds();
}
/**
 *
 * @param t2 The other Time object to compare with this one.
 * @return true if this Time object is behind the other one.
 */
public boolean isBehind( Time t2 ){
    return this.timeToSeconds() < t2.timeToSeconds();
}
/**
 *
 * @param t2 The other Time object to compare with this one.
 * @return true if this Time object represents a time
 * similar to this one.
 */
public boolean isSimilarTo( Time t2 ){
    return this.timeToSeconds() == t2.timeToSeconds();
}


    /**
     *
     * @return the seconds format of the input time.
     * e.g 00:01:00:00 becomes 3600
     */
        public long timeToSeconds(){
            return 86400*days+3600 * hours + 60 * minutes + seconds;
        }//end method

/**
 *
 * @return a String object representing this Time
 * object in dd:hh:mm:ss format.
 */
public String timeString(){
    return convertSecondsToTime(timeToSeconds());
}
        /**
         *
         * @return the system time and writes it in the hh:mm:ss format
         */
        public static String timeGetter() {//tells the time of day
            Calendar myCal = Calendar.getInstance();
            return STRING.purifier(String.format("%1$tH:%1$tM:%1$tS\n", myCal));
          }//end method


//converts raw seconds to the 00:00:00:00 format i.e days:hours:minutes:seconds format
/**
 *
 * @param seconds the time in seconds.
 * @return the time in the dd:hh:mm:ss format.
 */
public static String convertSecondsToTime( long seconds ){

long d = seconds/86400;
long h = (seconds%86400)/3600;
long m = ( seconds - (86400*d + 3600*h) )/60;
long s = ( seconds - (86400*d + 3600*h + 60*m ) );

String dd = String.valueOf(d);
String hh = String.valueOf(h);
String mm = String.valueOf(m);
String ss = String.valueOf(s);


                if (dd.length() == 1) {
                    dd = "0" + dd;
                }
                if (hh.length() == 1) {
                    hh = "0" + hh;
                }
                if (mm.length() == 1) {
                    mm = "0" + mm;
                }
                if (ss.length() == 1) {
                    ss = "0" + ss;
                }




            return dd + ":" + hh + ":" + mm + ":" + ss;
}
public boolean isBehindSystemTime(){
    return isBehind( new Time(Time.timeGetter()) );
}
public boolean isSimilarToSystemTime(){
    return isSimilarTo( new Time(Time.timeGetter()) );
}

public boolean isAheadOfSystemTime(){
    return isAhead( new Time(Time.timeGetter()) );
}

    @Override
    public String toString() {
        return timeString();
    }




    public static void main(String args[]){
        Time t1=new Time("03:10:45:22");
        Time t2=new Time("00:00:10:42");
System.out.println(t1.isSimilarTo(t2));
    }

    }//end inner class Timed