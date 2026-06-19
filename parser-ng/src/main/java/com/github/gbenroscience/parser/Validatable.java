/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package com.github.gbenroscience.parser;


import com.github.gbenroscience.util.ErrorLog;
import java.util.List;

/**
 *
 * @author JIBOYE OLuwagbemiro Olaoluwa
 */
public interface Validatable {

public boolean validate(List<String> scan, ErrorLog errorLog);

}