/*******************************************************************************
 * Copyright (C) 2017 Francois Petitjean
 * Contributors:
 * 	Francois Petitjean
 *      Germain Forestier
 * 
 * This file is part of 
 * "Generating synthetic time series to augment sparse datasets."
 * accepted for publication at the IEEE ICDM 2017 conference
 * 
 * "Generating synthetic time series to augment sparse datasets"
 * is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * "Generating synthetic time series to augment sparse datasets"
 * is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with "Generating synthetic time series to augment sparse datasets".
 * If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

import java.awt.Color;
import java.awt.Paint;

//import org.jfree.chart.renderer.xy.XYShapeRenderer;

public class Tools {
	public final static double Min3(final double a, final double b, final double c) {
		return (a <= b) ? ((a <= c) ? a : c) : (b <= c) ? b : c;
	}

	public static int ArgMin3(final double a, final double b, final double c) {
		return (a <= b) ? ((a <= c) ? 0 : 2) : (b <= c) ? 1 : 2;
	}
	
	public static double sum(final double... tab) {
		double res = 0.0;
		for (double d : tab)
			res += d;
		return res;
	}
	
	public static double max(final double... tab) {
		double max = Double.NEGATIVE_INFINITY;
		for (double d : tab){
			if(max<d){
				max = d;
			}
		}
		return max;
	}
	
	public static double min(final double... tab) {
		double min = Double.POSITIVE_INFINITY;
		for (double d : tab){
			if(d<min){
				min = d;
			}
		}
		return min;
	}
}
