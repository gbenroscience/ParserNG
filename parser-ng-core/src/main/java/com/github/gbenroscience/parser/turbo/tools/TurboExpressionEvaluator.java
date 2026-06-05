/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.parser.turbo.tools;
/**
 *
 * @author GBEMIRO
 * Interface for turbo compilers that generate optimized bytecode expressions.
 * Different implementations can be used for scalar, matrix, or hybrid operations.
 */
public interface TurboExpressionEvaluator {
    
    /**
     * Compile a postfix token array into a fast-executing expression.
     * 
     * @return A FastCompositeExpression ready for evaluation
     * @throws Throwable if compilation fails
     */
    FastCompositeExpression compile() throws Throwable;
}