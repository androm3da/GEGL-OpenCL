/* !!!! AUTOGENERATED FILE generated by svg-colormatrix.sh !!!!!  
 *                                                                
 *  Copyright 2006 Geert Jordaens <geert.jordaens@telenet.be>     
 *                                                                
 * !!!! AUTOGENERATED FILE !!!!!                                  
 *                                                                
 */
#if GEGL_CHANT_PROPERTIES
  gegl_chant_string (values, "", "list of <number>s")
#else

#define GEGL_CHANT_POINT_FILTER
#define GEGL_CHANT_NAME          svg_saturate
#define GEGL_CHANT_DESCRIPTION   "SVG color matrix operation svg_saturate"
#define GEGL_CHANT_CATEGORIES    "compositors:svgfilter"
#define GEGL_CHANT_SELF          "svg_saturate.c"
#define GEGL_CHANT_INIT
#include "gegl-chant.h"

#include <math.h>
#include <stdlib.h>

static void init (GeglChantOperation *self)
{
  GEGL_OPERATION_POINT_FILTER (self)->format = babl_format ("RaGaBaA float");
}

static gboolean
process (GeglOperation *op,
          void          *in_buf,
          void          *out_buf,
          glong          n_pixels)
{
  gfloat      *in = in_buf;
  gfloat      *out = out_buf;
  gfloat      *m;

  gfloat ma[25] = { 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0};
  char         *endptr;
  gfloat        value;
  const gchar   delimiter=',';
  const gchar  *delimiters=" ";
  gchar       **values;
  gint          i;

  m = ma;

  if ( GEGL_CHANT_OPERATION (op)->values != NULL ) 
    {
      g_strstrip(GEGL_CHANT_OPERATION (op)->values);      
      g_strdelimit (GEGL_CHANT_OPERATION (op)->values, delimiters, delimiter);
      values = g_strsplit (GEGL_CHANT_OPERATION (op)->values, &delimiter, 1);
      if ( values[0] != NULL )
        {
          value = g_ascii_strtod(values[0], &endptr);
          if (endptr != values[0])
            if ( value >= 0.0 && value <= 1.0 )
              {
                 ma[0]  = 0.213 + 0.787 * value;
                 ma[1]  = 0.715 - 0.715 * value;
                 ma[2]  = 0.072 - 0.072 * value;
                 ma[5]  = 0.213 - 0.213 * value;
                 ma[6]  = 0.715 + 0.285 * value;
                 ma[7]  = 0.072 - 0.072 * value;
                 ma[10] = 0.213 - 0.213 * value;
                 ma[11] = 0.715 - 0.715 * value;
                 ma[12] = 0.072 + 0.928 * value;
              }
        }
      g_strfreev(values);
    }
  for (i=0; i<n_pixels; i++)
    {
      out[0] =  m[0]  * in[0] +  m[1]  * in[1] + m[2]  * in[2] + m[3]  * in[3] + m[4];
      out[1] =  m[5]  * in[0] +  m[6]  * in[1] + m[7]  * in[2] + m[8]  * in[3] + m[9];
      out[2] =  m[10] * in[0] +  m[11] * in[1] + m[12] * in[2] + m[13] * in[3] + m[14];
      out[3] =  m[15] * in[0] +  m[16] * in[1] + m[17] * in[2] + m[18] * in[3] + m[19];
      in  += 4;
      out += 4;
    }
  return TRUE;
}

#endif
