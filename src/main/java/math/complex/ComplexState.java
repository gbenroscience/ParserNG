/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.complex;

/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public enum ComplexState {
  CARTESIAN,POLAR,EXPONENTIAL;

public boolean isCartesian(){
      return this == CARTESIAN;
  }
public boolean isPolar(){
      return this == POLAR;
  }
public boolean isExponential(){
    return this == EXPONENTIAL;
}

}
