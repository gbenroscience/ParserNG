/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import java.util.ArrayList;
import java.util.List;
import parser.Bracket;
import static parser.Number.*;
import static parser.Operator.*;
import static parser.Variable.*;
import static parser.methods.Method.*;

import java.util.logging.Level;
import java.util.logging.Logger;
import static math.differentialcalculus.Utilities.*;

/**
 *
 *
 * Objects of this class break down a scanned
 * function into simple format to which a simple
 * chain rule can be applied,following all the
 * principles of differentiation.
 *
 * For instance..4*x/(5+sin(4*x^2-7)) may be broken down into
 *
 * u1/(5+sin(u-7))...u1/(5+sin(v))....u1/(5+v1)...u1/v2
 *
 * At each level of simplification,a Differentiable object
 * is created and recorded which holds a reference to the
 * original function.
 *
 * @author GBEMIRO
 */
public class DerivativeStructureBuilder {

	private DifferentiableManager manager = new DifferentiableManager();

	/**
	 *
	 * @param expression The input expression to be differentiated.
	 * @throws Exception
	 */
	@SuppressWarnings("unused")
	public DerivativeStructureBuilder(String expression) throws Exception {
		try{
			CodeGenerator codeGenerator = new CodeGenerator(expression);
			ArrayList<String> scanner = new ArrayList<>(codeGenerator.getScanner());



			while(scanner.indexOf(")")!=-1){
				List<String>subList=null;
				try{
					//ty
					int closeindex = scanner.indexOf(")");

					int openindex = Bracket.getComplementIndex(false, closeindex, scanner);
					subList = scanner.subList(openindex+1, closeindex);
					print("---subList---before simplification---"+subList);
					if(subList.isEmpty()){break;}
					Formula.simplify(subList);
					print("---subList---after simplification---"+subList);
					handleSingleNumOrVar:{
						print("---handleSingleNumOrVar---"+subList);
						if(subList.size()==3 && isOpeningBracket(subList.get(0)) && isClosingBracket(subList.get(2)) ){
							if(!isVariableString(scanner.get(openindex-1))  ){
								subList.remove(0);subList.remove(2);
								Differentiable diff = new Differentiable(generateName(), new ArrayList<>(subList));
								manager.add(diff);
								subList.clear();
								subList.add(diff.getName());
							}

						}
						if(subList.size()==1&&(isNumber(subList.get(0))||isVariableString(subList.get(0)))){
							Differentiable diff = new Differentiable(generateName(), new ArrayList<>(subList));
							manager.add(diff);
							subList.clear();
							subList.add(diff.getName());
						}

					}//end block

					//print("---Single Number or Variable Block---scanner---"+scanner);
					handleInverse$Square$CubeOps:{
						for(int i=0;i<subList.size();i++){
							String op = subList.get(i);
							//3*x*x+2*x-¹*x^2-4/x
							if( (isInverse(op)||isSquare(op)||isCube(op)) && i>0){

								List <String>sub = subList.subList(i-1, i+1);
								Differentiable diff = new Differentiable(generateName(), new ArrayList<>(sub));
								manager.add(diff);
								sub.clear();
								sub.add(diff.getName());
								i-=1;
							}
						}//end for loop
					}//end block
					// print("----Inverse Block---scanner---"+scanner);
					handlePowerOp:{
						for(int i=0;i<subList.size();i++){
							String op = subList.get(i);
							//3*x*x+2*x-¹*x^2-4/x
							if(isPower(op)&&i>0){
								List <String>sub = subList.subList(i-1, i+2);
								Differentiable diff = new Differentiable(generateName(), new ArrayList<>(sub));
								manager.add(diff);
								sub.clear();
								sub.add(diff.getName());
								i-=2;
							}
						}//end for loop
					}//end block
					//print("--Power Block---scanner---"+scanner);
					handleMul$DivOps:{
						for(int i=0;i<subList.size();i++){
							String op = subList.get(i);
							//3*x*x+2*x-¹*x^2-4/x
							if(isMulOrDiv(op)&&i>0){
								List <String>sub = subList.subList(i-1, i+2);
								Differentiable diff = new Differentiable(generateName(), new ArrayList<>(sub));
								manager.add(diff);
								sub.clear();
								sub.add(diff.getName());
								i-=2;
							}
						}//end for loop
					}//end block
					//print("----Mul$Div Block---scanner---"+scanner);
					handleAdd$SubOps:{
						for(int i=0;i<subList.size();i++){
							String op = subList.get(i);
							//3*x*x+2*x-¹*x^2-4/x
							if(isPlusOrMinus(op)&&i>0){
								List <String>sub = subList.subList(i-1, i+2);
								Differentiable diff = new Differentiable(generateName(), new ArrayList<>(sub));
								manager.add(diff);
								sub.clear();
								sub.add(diff.getName());
								i-=2;
							}
						}//end for loop
					}//end block
					//print("---Add$Sub Block----scanner---"+scanner);

					if(openindex==0){
						closeindex = scanner.indexOf(")");

						openindex = Bracket.getComplementIndex(false, closeindex, scanner);
						List<String>sub = scanner.subList(openindex, closeindex+1);
						Differentiable diff =
								new Differentiable(generateName(), new ArrayList<>(sub.subList(1, sub.size()-1)));
						manager.add(diff);
						sub.clear();
						sub.add(diff.getName());
					}
					if(openindex>0&&isInBuiltMethod(scanner.get(openindex-1))){
						closeindex = scanner.indexOf(")");

						openindex = Bracket.getComplementIndex(false, closeindex, scanner);
						Differentiable diff =
								new Differentiable(generateName(), new ArrayList<>(scanner.subList(openindex-1, closeindex+1)));
						manager.add(diff);
						List<String>sub = scanner.subList(openindex-1, closeindex+1);sub.clear();
						sub.add(diff.getName());
					}


					else if(openindex>0&&!isInBuiltMethod(scanner.get(openindex-1))){
						closeindex = scanner.indexOf(")");

						openindex = Bracket.getComplementIndex(false, closeindex, scanner);
						Differentiable diff =
								new Differentiable(generateName(), new ArrayList<>(scanner.subList(openindex, closeindex+1)));
						manager.add(diff);
						List<String>sub = scanner.subList(openindex, closeindex+1);sub.clear();
						sub.add(diff.getName());
					}
					// print("--After final bracket operations---"+scanner);
    	/*  if(scanner.size()==1||scanner.size()==3){
  			 Differentiable diff =
  				    new Differentiable(generateName(), new ArrayList<>(scanner));
  			 manager.add(diff);
  			 scanner.clear();
  			 scanner.add(diff.getName());
    	  }
    		 print("--After final clean-up operations---"+scanner);
           */

					//print("diff tree---"+manager.getDIFFERENTIABLES());
				}//end try
				catch(IndexOutOfBoundsException boundsException){
					boundsException.printStackTrace();
					break;
				}//end catch
			}//end while loop
			//print("Finished tree structure build!----scanner---"+scanner );




		}//end try
		catch (Exception ex) {
			Logger.getLogger(CodeGenerator.class.getName()).log(Level.SEVERE, null, ex);
			throw new Exception("Bad Input!");
		}//end catch




	}//end constructor.

	public DifferentiableManager getManager() {
		return manager;
	}





	/**
	 * Automatically generates a name for
	 * a given Differentiable object..especially
	 * since these are mostly automatically created on the
	 * fly during differential parsing.
	 * @return a unique name for the object.
	 */
	private String generateName(){
		int count = manager.count();
		return "myDiff_"+count;
	}//end method

























	public static void main(String args[]){
		try {

			String expression = //"8-(5+2+6)-2+6+1*2-x*y*3*(2*x*x*2*x*x*8*x*7*x*4*y+(3*2*8+4/21*x)-¹-(((5*2)))-3*x*x^2*2*3*4^x-5*sin(1/x))+5*3/6*8+4*x*3";
					"3*x/(5-sin(4*x^2-7))+7*x^2";
			DerivativeStructureBuilder builder = new DerivativeStructureBuilder(expression);


		} catch (Exception ex) {
			Logger.getLogger(DerivativeStructureBuilder.class.getName()).log(Level.SEVERE, null, ex);
		}

	}//end method main



}//end class