# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# '@' means "wall"
# 'P' means "player" spawn point
# 'B' means green apple spawn point
# 'R' means red apple spawn point
# '' is empty space

HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P AAAAA    P  A P  @',
    '@  P     A P AA    P    AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A      P @',
    '@A  AAA  A  A  AAA A    AAAA     P   @',
    '@    A A   AAA  A A      A AA   A  P @',
    '@     AAA   A A  AAA      AA   AAA P @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A         P  P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

NORM_MAP = [
    '@@@@@@@@@@@@@@@@@@',
    '@GGBGG   B  GBGGR@',
    '@GRRG  B   R GGGG@',
    '@GGGGG R    GGGGG@',
    '@RGGG   P  R GGRG@',
    '@GBGGGR   P GBBGG@',
    '@GGBG   R R  BGRG@',
    '@RRGGGB     GBRGG@',
    '@GBGG   B  R GGBG@',
    '@GGBG R      GRGG@',
    '@GRGG   RP P BGGB@',
    '@GGBGR   P  GGRGG@',
    '@BGGG  R   P BGGG@',
    '@GBRG   P   GGRBG@',
    '@GRGB    R   GGBG@',
    '@GBBGRR  P  BGGGG@',
    '@GGGB  R     GBGG@',
    '@GBGGG  P P BGGRG@',
    '@GRGB     R  GGRR@',
    '@BBGGG  R   GBBGG@',
    '@GGGB R      GRGG@',
    '@GBGGG  B   BGRRG@',
    '@GGGB    R   GGGB@',
    '@GGRGB    B GBGGG@',
    '@@@@@@@@@@@@@@@@@@']




