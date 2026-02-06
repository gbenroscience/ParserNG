package logic;

public enum BASE_MODE {

        BIN, OCT, DEC;

        /**
         * @return the number base from the {@link CalcLogic#baseMode value}
         */
        public final int getBase() {
            switch (this) {
                case BIN:
                    return 2;
                case OCT:
                    return 8;
                case DEC:
                    return 10;
                default:
                    return 10;

            }
        }

}
