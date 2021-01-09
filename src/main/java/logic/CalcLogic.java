package logic;
 
import interfaces.Savable;

import java.util.ArrayList;

import util.FormulaeManager;
import util.HistoryManager;
import util.MathExpressionManager;
import util.Mode;
import util.Serializer;
import util.Settings;


public class CalcLogic implements Savable{
    
    /**
     * Stores the commands used on the CommandLineActivity
     */
    private final ArrayList<String> commandHistory = new ArrayList<>();
    
    private final HistoryManager histMan =  new HistoryManager();
    
    private final MathExpressionManager funcMan = new MathExpressionManager();

    private final FormulaeManager formMan = new FormulaeManager();

    private final Mode mode = Mode.CALCULATOR;// The usage MODE of this tool..either via the command line or in the calculator MODE.
    
    /**
     * When true, the calculator is in a state wherein the last action it performed
     * was the evaluation of an expression.
     */
    private boolean justEvaluated = false;


    private Settings settings = new Settings();

    private OperatingSystem manager;
    private boolean vibrateOn = true;
    private BASE_MODE baseMode = BASE_MODE.DEC;

    private TRIG_MODE trigMode = TRIG_MODE.NORMAL;


    private DRG_MODE drgMode = DRG_MODE.DEG;
    private POWER_MODE powerMode = POWER_MODE.ON;

    private USE_MODE useMode = USE_MODE.NORMAL;


    private boolean running = true;




    public CalcLogic(){

    }

    public void setRunning(boolean running) {
        this.running = running;
    }

    public boolean isRunning() {
        return running;
    }



    public ArrayList<String> getCommandHistory() {
        return commandHistory;
    }



    public HistoryManager getHistMan() {
        return histMan;
    }

    public MathExpressionManager getFuncMan() {
        return funcMan;
    }

    public FormulaeManager getFormMan() {
        return formMan;
    }

    public Mode getMode() {
        return mode;
    }

    public void setJustEvaluated(boolean justEvaluated) {
        this.justEvaluated = justEvaluated;
    }

    public boolean isJustEvaluated() {
        return this.justEvaluated;
    }

    public Settings getSettings() {
        return settings;
    }

    public void setSettings(Settings settings) {
        this.settings = settings;
    }

    public  void setManager(OperatingSystem manager) {
        this.manager = manager;
    }

    public OperatingSystem getManager() {
        return this.manager;
    }

    public void setVibrateOn(boolean vibrateOn) {
        this.vibrateOn = vibrateOn;
    }

    public boolean isVibrateOn() {
        return vibrateOn;
    }

    public void setBaseMode(BASE_MODE baseMode) {
        this.baseMode = baseMode;
    }

    public BASE_MODE getBaseMode() {
        return baseMode;
    }

    public int getBase(){
        return baseMode.getBase();
    }

    public void setTrigMode(TRIG_MODE trigMode) {
        this.trigMode = trigMode;
    }

    public TRIG_MODE getTrigMode() {
        return this.trigMode;
    }

    public void setDrgMode(DRG_MODE drgMode) {
        this.drgMode = drgMode;
    }

    public DRG_MODE getDrgMode() {
        return drgMode;
    }

    public void setPowerMode(POWER_MODE powerMode) {
        this.powerMode = powerMode;
    }

    public POWER_MODE getPowerMode() {
        return powerMode;
    }

    public void setUseMode(USE_MODE useMode) {
        this.useMode = useMode;
    }

    public USE_MODE getUseMode() {
        return useMode;
    }

    @Override
    public String serialize() {
        return Serializer.serialize(this);
    }

    
    public static CalcLogic parse(String enc) {
        return (CalcLogic) Serializer.deserialize(enc);
    }


}
