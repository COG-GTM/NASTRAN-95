C     SHARED-KERNEL PARITY DRIVER. READS LIST-DIRECTED INPUT ON STDIN:
C       NMAT1, THEN NMAT1 LINES: MID E G NU RHO ALPHA TREF GE ST SC SS
C       NMAT2, THEN NMAT2 LINES: MID G11 G12 G13 G22 G23 G33 RHO A1 A2
C                                A12 TREF GE ST SC SS
C       NCSTM, THEN NCSTM LINES: CID TYPE X0 Y0 Z0 T11 ... T33
C       NCASE, THEN NCASE LINES: KIND ...
C         KIND 1 (MAT)    : INFLAG MATID SINTH COSTH -> /MATOUT/ 1-25
C         KIND 2 (TRANSS) : CID X Y Z                -> TA(9) SINGLE
C         KIND 3 (TRANSD) : CID X Y Z                -> TA(9) DOUBLE
C     ONE OUTPUT LINE PER CASE.
C
      PROGRAM KERNEL
      INTEGER          IZ(300000),BFR(10),N2MAT,KIND
      REAL             R(16),ECPT(4),TA(9)
      DOUBLE PRECISION TAD(9)
      INTEGER          IECPT(4)
      CHARACTER*512    LINE
      COMMON /ZZZZZZ/  Z(300000)
      COMMON /SYSTEM/  ISYS(100)
      COMMON /NAMES /  RD,RDREW,WRT,WRTREW,CLSREW
      INTEGER          RD,RDREW,WRT,WRTREW,CLSREW
      COMMON /MATIN /  MATID,INFLAG,TEMP,PLAARG,SINTH,COSTH
      COMMON /MATOUT/  OUT(25),ITDEP,IDUM(26)
      COMMON /HSTUB /  NM1,M1(12,200),NM2,M2(17,200),ICARD,IPOS
      EQUIVALENCE      (Z(1),IZ(1)),(ECPT(1),IECPT(1))
      PARAMETER        (ICSTM = 200001)
C
      ISYS(1) = 1000
      ISYS(2) = 6
      ISYS(10)= 0
      RD = 2
      RDREW = 0
      WRT = 3
      WRTREW = 1
      CLSREW = 1
C
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
      READ (*,*) NCS
      DO 30 I = 1,NCS
      K = ICSTM + 14*(I-1)
      READ (*,*) IZ(K),IZ(K+1),(Z(K+J),J=2,13)
   30 CONTINUE
C
      CALL PREMAT (IZ(1),Z(1),BFR,100000,N2MAT,101,0)
      CALL PRETRS (Z(ICSTM),14*NCS)
      CALL PRETRD (Z(ICSTM),14*NCS)
C
      READ (*,*) NCASE
      DO 100 IC = 1,NCASE
      READ (*,'(A)') LINE
      READ (LINE,*) KIND
      IF (KIND .EQ. 1) THEN
         READ (LINE,*) KIND,INFLAG,MATID,SINTH,COSTH
         TEMP = 0.0
         CALL MAT (1)
         WRITE (*,'(25(1X,ES16.8E3))') (OUT(J),J=1,25)
      ELSE
         READ (LINE,*) KIND,IECPT(1),(ECPT(J),J=2,4)
         IF (KIND .EQ. 2) THEN
            CALL TRANSS (ECPT,TA)
            WRITE (*,'(9(1X,ES16.8E3))') TA
         ELSE
            CALL TRANSD (ECPT,TAD)
            WRITE (*,'(9(1X,ES25.16E3))') TAD
         END IF
      END IF
  100 CONTINUE
      END
