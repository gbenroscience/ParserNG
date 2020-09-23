/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.complex;


/**
 *
 * Objects of this class model complex numbers.
 * Before creation, the object data could be in polar, rectangular(cartesian) or exponential
 * form, but once created, all states revert to cartesian.
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class ComplexNumber {
    private double real;
    private double imag;
    public static final String radical = "i";
    private ComplexState state = ComplexState.CARTESIAN;
    /**
     * The 2 double arguments have interpretations that vary with the state argument.
     * If state = POLAR ,
     * then the first double argument represents the radius or 
     * absolute value of the ComplexNumber and the second double argument represents its angle.
     *
     * If state = CARTESIAN , then the first double argument represents the real part of the complex number
     * and the second one represents its imaginary part.
     *
     * If state = EXPONENTIAL, then the first argument represents the radius or
     * absolute value of the ComplexNumber and the second double argument represents its angle just as in
     * the Polar form.
     *
     * @param real The real part in CARTESIAN and the radius in POLAR.
     * @param imag The imaginary part in CARTESIAN and the angle in POLAR
     * @param state The state of the ComplexNumber object, POLAR or CARTESIAN.
     */
    public ComplexNumber(double real, double imag,ComplexState state) {
            this.state = state;
        if (state.isCartesian()){
        this.real = real;
        this.imag = imag;
        }//end if
        else if(state.isPolar()||state.isExponential()){
            this.real = real*Math.cos(imag);
            this.imag = real*Math.sin(imag);
        }//end else
//revert all states back to cartesian the moment
//proper conversion has been done.
this.state = ComplexState.CARTESIAN;
    }//end constructor



    public double getImag() {
        return imag;
    }

    public void setImag(double imag) {
        this.imag = imag;
    }

    public double getReal() {
        return real;
    }

    public void setReal(double real) {
        this.real = real;
    }

    public static String getRadical() {
        return radical;
    }

    public void setState(ComplexState state) {
        this.state = state;
    }

    public ComplexState getState() {
        return state;
    }
/**
 *
 * @return the angle of this object
 */
    public double getAngle(){
        if(state.isCartesian()){
            return Math.atan(imag/real);
        }
        else{
            return imag;
        }
    }//end method
 /**
 *
 * @return the radius of this object
 */
    public double getRadius(){
        if(state.isPolar()){
            return abs();
        }
        else{
            return real;
        }
    }//end method




/**
 * Adds two ComplexNumber objects
 * @param complexNumber The ComplexNumber object to add to this one.
 * @return the sum of the two objects as a new ComplexNumber object.
 */
    public ComplexNumber add(ComplexNumber complexNumber){
        return new ComplexNumber(real+complexNumber.real, imag+complexNumber.imag,state);
    }
/**
 * Subtracts the parameter ComplexNumber object from this ComplexNumber object.
 * @param complexNumber The ComplexNumber object to subtract from this one.
 * @return the difference of the two objects as a new ComplexNumber object.
 */
    public ComplexNumber minus(ComplexNumber complexNumber){
        return new ComplexNumber(real-complexNumber.real, imag-complexNumber.imag,state);
    }

/**
 * Multiplies this ComplexNumber object by the parameter ComplexNumber object.
 * @param complexNumber The ComplexNumber object to be employed in multiplying this one.
 * @return the product of the two objects as a new ComplexNumber object.
 */
public ComplexNumber multiply(ComplexNumber complexNumber){
double a = real;
double b =imag;

double c =complexNumber.real;
double d =complexNumber.imag;
    return new ComplexNumber( (a*c-b*d) ,   (a*d + b*c),state);
}

/**
 * Divides this ComplexNumber object by the parameter ComplexNumber object.
 * @param complexNumber The ComplexNumber object to be employed in dividing this one.
 * @return the division of the two objects as a new ComplexNumber object.
 */
public ComplexNumber divide(ComplexNumber complexNumber){
double a = real;
double b =imag;

double c =complexNumber.real;
double d =complexNumber.imag;
    return new ComplexNumber( (a*c+b*c+b*d)/( c*c + d*d )  ,   (a*d)/( c*c + d*d ),state );
}



/**
 * Divides this ComplexNumber object by the number.
 * @param number The scalar to be used in dividing this object
 * @return a scaled version of this object stored in a new object.
 */
public ComplexNumber scalarDivide(double number){
    return new ComplexNumber( real/number,imag/number,ComplexState.CARTESIAN );
}

/**
 * Multiplies this ComplexNumber object by the number.
 * @param number The scalar to be used in multiplying this object
 * @return a magnified version of this object stored in a new object.
 */
public ComplexNumber scalarMultiply(double number){
    return new ComplexNumber( real*number,imag*number,ComplexState.CARTESIAN );
}


/**
 *
 * @return the conjugate of this object.
 */
public ComplexNumber getConjugate(){
    return new ComplexNumber(real, -imag,ComplexState.CARTESIAN);
}


/**
 * Finds the absolute value of this object
 * @return the absolute value of this objet.
 */
public double abs(){
double a = real;
double b =imag;
return Math.sqrt(a*a+b*b);
}
/**
 *
 * @return the natural logarithm of a ComplexNumber object and stores the result in a new object.
 */
public ComplexNumber log(){
    return new ComplexNumber( Math.log(getRadius()), getAngle(), ComplexState.CARTESIAN);
}

/**
 *
 * @return the exponent of a CompleNumber object(e^Z) as a new ComplexNumber object.
 */
public ComplexNumber exp(){
    return new ComplexNumber( Math.exp(real)*Math.cos(imag), Math.exp(real)*Math.sin(imag), ComplexState.CARTESIAN);
}
/** 
 * 
 * @return the inverse of this object
 */
public ComplexNumber inverse(){
    double den = real*real + imag*imag;
    return new ComplexNumber(real/den, -imag/den, ComplexState.CARTESIAN);
}


/**
 *
 * @param n The power to raise the ComplexNumber object to.
 * @return the nth power of the ComplexNumber object.
 */

public ComplexNumber pow(double n){
    double r = abs();
    return new ComplexNumber(  Math.pow( r,n ) , n*getAngle(), ComplexState.POLAR);
}
/**
 *
 * @return the sin of the ComplexNumber object.
 */

public ComplexNumber sin(){
    return new ComplexNumber( Math.sin(real)*Math.cosh(imag), Math.cos(real)*Math.sinh(imag), ComplexState.CARTESIAN);
}
/**
 *
 * @return the cosine of the ComplexNumber object.
 */

public ComplexNumber cos(){
    return new ComplexNumber( Math.cos(real)*Math.cosh(imag), -Math.sin(real)*Math.sinh(imag), ComplexState.CARTESIAN);
}


/**
 *
 * @return the tangent of this ComplexNumber object
 */
public ComplexNumber tan(){
    double tanreal = Math.tan(real);
    double tanhimag = Math.tanh(imag);

   double sechImag = 1/Math.cosh(imag);
   double secReal = 1/Math.cos(real);

   double den = 1 - tanreal*tanreal*tanhimag*tanhimag;

return new ComplexNumber( tanreal*sechImag*sechImag/den  ,  tanhimag*secReal*secReal/den , ComplexState.CARTESIAN);
}
/**
 *
 * @return the secant of this ComplexNumber object
 */
public ComplexNumber sec(){
    return cos().inverse();
}
/**
 *
 * @return the cosecant of this ComplexNumber object
 */
public ComplexNumber csc(){
    return sin().inverse();
}

/**
 *
 * @return the cotangent of this ComplexNumber object
 */
public ComplexNumber cot(){
    return tan().inverse();
}


/**
 *
 * @return the hyperbolic sine of this ComplexNumber object
 */
public ComplexNumber sinh(){
    return new ComplexNumber( Math.sinh(real)*Math.cos(imag),   Math.cosh(real)*Math.sin(imag), ComplexState.CARTESIAN);
}

/**
 *
 * @return the hyperbolic cosine of this ComplexNumber object
 */
public ComplexNumber cosh(){
    return new ComplexNumber( Math.cosh(real)*Math.cos(imag),   Math.sinh(real)*Math.sin(imag), ComplexState.CARTESIAN);
}


/**
 *
 * @return the hyperbolic tangent of this ComplexNumber object
 */
public ComplexNumber tanh(){
   double tanhreal = Math.tanh(real);
    double tanimag = Math.tan(imag);

   double secImag = 1/Math.cos(imag);
   double sechReal = 1/Math.cosh(real);

   double den = 1 + tanhreal*tanhreal*tanimag*tanimag;

return new ComplexNumber( tanhreal*secImag*secImag/den  ,  -tanimag*sechReal*sechReal/den , ComplexState.CARTESIAN);

}



/**
 *
 * @return the hyperbolic secant of this ComplexNumber object
 */
public ComplexNumber sech(){
    return cosh().inverse();
}
/**
 *
 * @return the hyperbolic cosecant of this ComplexNumber object
 */
public ComplexNumber csch(){
    return sinh().inverse();
}

/**
 *
 * @return the hyperbolic cotangent of this ComplexNumber object
 */
public ComplexNumber coth(){
    return tanh().inverse();
}


    @Override
    public String toString() {
     String comp = real + "+" +imag+radical;

     comp = comp.replace("+-", "-");
     comp = comp.replace("-+", "-");
     comp = comp.replace("--", "+");
     comp = comp.replace("++", "+");
return comp;
    }





















    public static void main(String args[]){

 ComplexNumber Z = new ComplexNumber(2, -3.5, ComplexState.CARTESIAN);
 ComplexNumber Z1 = new ComplexNumber(2, -3.5, ComplexState.CARTESIAN);

System.out.println( Z.sin().pow(2).add( Z.cos().pow(2) ) );
    }



}