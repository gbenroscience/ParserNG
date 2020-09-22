package logic;
 

import java.util.ArrayList;

import interfaces.Doable;
import util.Log;

/**
 * Objects of this class will control the running modes of the calculator.
 * And so enable the user to use a single button such as the equals button to act on many kinds of input.
 * e.g calculate,integrate,solve equations,store objects and so on.
 * This class will be involved with the different threads that may be spawned by the software in future.
 * @author GBEMIRO
 */
public class OperatingSystem {
    private Doable task;//task executor

    /**
     *
     * @return the Task object
     */
    public Doable getTask() {
        return task;
    }
    /**
     *
     * @param task sets the Task object
     */
    public void setTask(Doable task) {
        this.task = task;
    }


    /**
     * method responsible for recognizing commands entered into the command line or text field and executing them.
     * @param task the command to be executed.
     */
    public void execute(CalcLogic calcLogic,final Doable task){
        try{
            if(calcLogic.getMode().isCalculator()){
                task.eval();
            }//end if
            else{
                Log.e("Error", "Unrecognized Operation!");
            }

        }//end try

        catch(NullPointerException nolian){
        }//end catch
        catch(IndexOutOfBoundsException indexErr){

        }//end catch








    }//end method execute





}//end class OperatingSystem