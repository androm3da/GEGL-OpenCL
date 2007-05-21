/* A spatial color to greyscale converter.
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * GEGL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GEGL; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 *
 * Copyright 2007 Øyvind Kolås <oeyvindk@hig.no>
 *                Ivar Farup   <ivarf@hig.no>
 */

#if GEGL_CHANT_PROPERTIES 

gegl_chant_int (radius,     2, 5000.0, 384, "neighbourhood taken into account")
gegl_chant_int (samples,    0, 1000,    3,  "number of samples to do")
gegl_chant_int (iterations, 0, 1000.0, 23, "number of iterations (length of exposure)")
gegl_chant_double (strength,  -8, 8,  0.5, "how much the local optimum separation should be taken into account.")
gegl_chant_double (gamma, 0.0, 10.0, 1.6, "post correction gamma.")
#else

#define GEGL_CHANT_FILTER
#define GEGL_CHANT_NAME         c2g
#define GEGL_CHANT_DESCRIPTION  "Color to grayscale conversion that uses, spatial color differences to perform local grayscale contrast enhancement."
#define GEGL_CHANT_SELF         "c2g.c"
#define GEGL_CHANT_CATEGORIES   "enhance"
#define GEGL_CHANT_CLASS_INIT
#include "gegl-chant.h"
#include <math.h>
#define sq(a) ((a)*(a))

static void c2g (GeglBuffer *src,
                 GeglBuffer *dst,
                 gint        radius,
                 gint        samples,
                 gint        iterations,
                 gfloat      strength,
                 gfloat      gamma);

static GeglRectangle get_source_rect (GeglOperation *self,
                                      gpointer       context_id);

static void compute_luts(void);
#include <stdlib.h>

static gboolean
process (GeglOperation *operation,
         gpointer       context_id)
{
  GeglOperationFilter *filter;
  GeglChantOperation  *self;
  GeglBuffer          *input;
  GeglBuffer          *output;


  filter = GEGL_OPERATION_FILTER (operation);
  self   = GEGL_CHANT_OPERATION (operation);


  input = GEGL_BUFFER (gegl_operation_get_data (operation, context_id, "input"));
  {
    GeglRectangle   *result = gegl_operation_result_rect (operation, context_id);
    GeglRectangle    need   = *result;
    GeglBuffer      *temp_in;

    need.x-=self->radius;
    need.y-=self->radius;
    need.width +=self->radius*2;
    need.height +=self->radius*2;

    if (result->width == 0 ||
        result->height== 0 ||
        self->radius < 1.0)
      {
        output = g_object_ref (input);
      }
    else
      {
        temp_in = g_object_new (GEGL_TYPE_BUFFER,
                               "source", input,
                               "x",      need.x,
                               "y",      need.y,
                               "width",  need.width,
                               "height", need.height,
                               NULL);

        output = g_object_new (GEGL_TYPE_BUFFER,
                               "format", babl_format ("RGBA float"),
                               "x",      need.x,
                               "y",      need.y,
                               "width",  need.width,
                               "height", need.height,
                               NULL);

        c2g (temp_in, output, self->radius, self->samples, self->iterations, self->strength, self->gamma);
        g_object_unref (temp_in);
      }

    {
      GeglBuffer *cropped = g_object_new (GEGL_TYPE_BUFFER,
                                          "source", output,
                                          "x",      result->x,
                                          "y",      result->y,
                                          "width",  result->width ,
                                          "height", result->height,
                                          NULL);
      gegl_operation_set_data (operation, context_id, "output", G_OBJECT (cropped));
      g_object_unref (output);
    }
  }
  return  TRUE;
}

#include "envelopes.h"

static void c2g (GeglBuffer *src,
                 GeglBuffer *dst,
                 gint        radius,
                 gint        samples,
                 gint        iterations,
                 gfloat      strength,
                 gfloat      gamma)
{
  gint x,y;
  gfloat *src_buf;
  gfloat *dst_buf;

  compute_luts ();

  src_buf = g_malloc0 (src->width * src->height * 4 * 4);
  dst_buf = g_malloc0 (dst->width * dst->height * 4 * 4);

  gegl_buffer_get (src, NULL, 1.0, babl_format ("RGBA float"), src_buf);
  for (y=radius; y<dst->height-radius; y++)
    {
      gint offset = ((src->width*y)+radius)*4;
      for (x=radius; x<dst->width-radius; x++)
        {
          gfloat  min_envelope[4];
          gfloat  max_envelope[4];
          gfloat *pixel = src_buf + offset;

          compute_envelopes (src_buf,
                             src->width,
                             src->height,
                             x, y,
                             radius, samples,
                             iterations,
                             min_envelope, max_envelope);

          { /* now having a local blackpoint and a local white point
               we just wonder whether we are more black than white */
            gfloat gray;
            gint   c;
          
            gray = pixel[0]*0.212671 + pixel[1] * 0.715160 + pixel[2] * 0.072169;
            {
              gfloat nominator = 0;
              gfloat denominator = 0; 
              for (c=0; c<3; c++)
                {
                  gfloat delta = max_envelope[c]-min_envelope[c];
                  nominator += (pixel[c] - min_envelope[c]) * delta;
                  denominator += delta*delta;
                }

              if (denominator>0.000) /* if we found a range, modify the result */
                {
                  gray *= (1.0-strength);

                  if (gamma==1.0)
                    {
                     gray += strength * (nominator/denominator);
                    }
                  else
                    {
                     gray += pow (strength * (nominator/denominator), gamma);
                    }

                }
            }

            for (c=0; c<3;c++)
              dst_buf[offset+c] = gray;
            dst_buf[offset+c] = pixel[c]; /* alpha is unmodified */
          }
          offset+=4;
        }
    }
  gegl_buffer_set (dst, NULL, babl_format ("RGBA float"), dst_buf);

  g_free (src_buf);
  g_free (dst_buf);
}
              
static GeglRectangle
get_defined_region (GeglOperation *operation)
{
  GeglRectangle  result = {0,0,0,0};
  GeglRectangle *in_rect = gegl_operation_source_get_defined_region (operation,
                                                                     "input");

  GeglChantOperation *blur = GEGL_CHANT_OPERATION (operation);
  gint       radius = ceil(blur->radius);

  if (!in_rect)
    return result;

  result = *in_rect;
  if(0)if (result.width  != 0 &&
      result.height  != 0)
    {
      result.x-=radius;
      result.y-=radius;
      result.width +=radius*2;
      result.height +=radius*2;
    }
  
  return result;
}

static GeglRectangle get_source_rect (GeglOperation *self,
                                      gpointer       context_id)
{
  GeglChantOperation *blur   = GEGL_CHANT_OPERATION (self);
  GeglRectangle            rect;
  gint                radius;
 
  radius = ceil(blur->radius);

  rect  = *gegl_operation_get_requested_region (self, context_id);
  if (rect.width  != 0 &&
      rect.height  != 0)
    {
      rect.x -= radius;
      rect.y -= radius;
      rect.width  += radius*2;
      rect.height  += radius*2;
    }

  return rect;
}

static gboolean
calc_source_regions (GeglOperation *self,
                     gpointer       context_id)
{
  GeglRectangle need = get_source_rect (self, context_id);

  gegl_operation_set_source_region (self, context_id, "input", &need);

  return TRUE;
}

static GeglRectangle
get_affected_region (GeglOperation *self,
                     const gchar   *input_pad,
                     GeglRectangle  region)
{
  GeglChantOperation *blur   = GEGL_CHANT_OPERATION (self);
  gint                radius;
 
  radius = ceil(blur->radius);

  region.x -= radius;
  region.y -= radius;
  region.width  += radius*2;
  region.height  += radius*2;
  return region;
}

static void class_init (GeglOperationClass *operation_class)
{
  operation_class->get_defined_region  = get_defined_region;
  operation_class->get_affected_region = get_affected_region;
  operation_class->calc_source_regions = calc_source_regions;
}

#endif
