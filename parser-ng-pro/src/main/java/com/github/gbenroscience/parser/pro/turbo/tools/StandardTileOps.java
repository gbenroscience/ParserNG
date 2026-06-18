package com.github.gbenroscience.parser.pro.turbo.tools;

/**
 *
 * @author GBEMIRO
 */
public enum StandardTileOps implements TileOperator {
    SIN {
        @Override
        public void evaluate(double[] src, double[] dest, int tileSize) {
            for (int i = 0; i < tileSize; i++) {
                dest[i] = Math.sin(src[i]); // Clean, unrollable loop
            }
        }
    }, 
    COS {
        @Override
        public void evaluate(double[] src, double[] dest, int tileSize) {
            for (int i = 0; i < tileSize; i++) {
                dest[i] = Math.cos(src[i]);
            }
        }
    },
    LOG {
        @Override
        public void evaluate(double[] src, double[] dest, int tileSize) {
            for (int i = 0; i < tileSize; i++) {
                dest[i] = Math.log(src[i]);
            }
        }
    }
    // Add all 50+ ParserNG methods here. 
    // They are completely isolated from each other!
}