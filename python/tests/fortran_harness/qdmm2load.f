C     CQDMEM2 THERMAL LOAD HARNESS DRIVER (MIS/QDMM2.F, AS CALLED FROM
C     EDTL).
C
C     STDIN:  NM1 / MAT1 LINES, NM2 / MAT2 LINES, NCS / CSTM LINES (SAME
C             FORMAT AS DRIVER.F), THEN NCASE AND FOR EACH CASE
C               KIND NAME N
C               IFLAG(1..N)     1 = INTEGER WORD, 0 = REAL WORD
C               VALUE(1..N)     QDMEM2 ECPT RECORD (26 WORDS)
C               TEMP NPG        ELEMENT TEMPERATURE TI(1), LOAD LENGTH
C
C     STDOUT: PER CASE 'CASE n KIND NAME', 'LOAD NPG' AND THE NPG
C             SIL-ORDERED LOAD WORDS (9 SIGNIFICANT DIGITS), 'END'.
C             BASGLB READS THE CSTM RECORDS THROUGH THE CSTMRD STUB.
C
      PROGRAM QD2LDH
      INTEGER          IZ(300000),BFR(10),N2MAT,KIND,N,IFLAG(400)
      REAL             R(16),WORDS(400),TI(2),PG(400)
      INTEGER          IWORDS(400)
      DOUBLE PRECISION XVAL(400)
      CHARACTER*8      NAME
      COMMON /ZZZZZZ/  Z(300000)
      COMMON /SYSTEM/  ISYS(100)
      COMMON /NAMES /  RD,RDREW,WRT,WRTREW,CLSREW
      INTEGER          RD,RDREW,WRT,WRTREW,CLSREW
      COMMON /HSTUB /  NM1,M1(12,200),NM2,M2(17,200),ICARD,IPOS
      COMMON /CONDAS/  CONSTS(5)
      COMMON /TRIMEX/  TRX(100)
      COMMON /LOADX /  LC(4),CSTM,LC1(10),IDUM(3),ICM
      INTEGER          CSTM
      COMMON /HLOAD /  TSTUB,DSTUB,NCSTM,ICS,ICSTM0
      EQUIVALENCE      (Z(1),IZ(1)),(WORDS(1),IWORDS(1))
      PARAMETER        (ICSTM = 200001)
      ISYS(1) = 1000
      ISYS(2) = 6
      ISYS(9) = 50
      ISYS(10)= 0
      ISYS(40)= 32
      RD = 2
      RDREW = 0
      WRT = 3
      WRTREW = 1
      CLSREW = 1
      CONSTS(1) = 4.0*ATAN(1.0)
      CONSTS(2) = 2.0*CONSTS(1)
      CONSTS(3) = 180.0/CONSTS(1)
      CONSTS(4) = CONSTS(1)/180.0
      CONSTS(5) = 4.0*CONSTS(1)**2
      READ (*,*) NM1
      DO 10 I = 1,NM1
      READ (*,*) M1(1,I),(R(J),J=1,10)
      DO 5 J = 1,10
    5 M1(J+1,I) = TRANSFER(R(J),0)
   10 M1(12,I) = 0
      READ (*,*) NM2
      DO 20 I = 1,NM2
      READ (*,*) M2(1,I),(R(J),J=1,15)
      DO 15 J = 1,15
   15 M2(J+1,I) = TRANSFER(R(J),0)
   20 M2(17,I) = 0
      READ (*,*) NCSTM
      DO 30 I = 1,NCSTM
      K = ICSTM + 14*(I-1)
      READ (*,*) IZ(K),IZ(K+1),(Z(K+J),J=2,13)
   30 CONTINUE
      ICSTM0 = ICSTM
      ICS = 0
      CALL PREMAT (IZ(1),Z(1),BFR,100000,N2MAT,101,0)
      CSTM = 204
      ICM  = 0
C
      READ (*,*) NCASE
      DO 900 IC = 1,NCASE
      READ (*,*) KIND,NAME,N
      READ (*,*) (IFLAG(J),J=1,N)
      READ (*,*) (XVAL(J),J=1,N)
      DO 40 J = 1,N
      IF (IFLAG(J) .EQ. 1) THEN
         IWORDS(J) = NINT(XVAL(J))
      ELSE
         WORDS(J) = REAL(XVAL(J))
      END IF
   40 CONTINUE
      READ (*,*) TI(1),NPG
      TI(2) = 0.0
      WRITE (*,'(A,2I6,1X,A)') 'CASE',IC,KIND,NAME
      DO 50 J = 1,100
   50 TRX(J) = 0.0
      DO 60 J = 1,N
   60 TRX(J) = WORDS(J)
      DO 70 J = 1,NPG
   70 PG(J) = 0.0
      CALL QDMM2 (TI,PG)
      WRITE (*,'(A,I8)') 'LOAD',NPG
      WRITE (*,'(6(1X,ES16.8E3))') (PG(J),J=1,NPG)
      WRITE (*,'(A)') 'END'
  900 CONTINUE
      END
