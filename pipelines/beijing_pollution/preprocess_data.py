from csv import DictWriter
from csv import excel
from io import StringIO
from datetime import datetime
import tempfile
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
# convert series to supervised learning


input_filename = 'pollution_raw.csv'
output_filename = 'pollution_preprocessed.csv'

def _dict_to_csv(element, column_order, missing_val='', discard_extras=True, dialect=excel):
    """ Additional properties for delimiters, escape chars, etc via an instance of csv.Dialect
        Note: This implementation does not support unicode
    """

    buf = StringIO()

    writer = DictWriter(
		buf,
        fieldnames=column_order,
        restval=missing_val,
        extrasaction=('ignore' if discard_extras else 'raise'),
        dialect=dialect
	)
    writer.writerow(element)

    return buf.getvalue().rstrip(dialect.lineterminator)


class _DictToCSVFn(beam.DoFn):
    """ Converts a Dictionary to a CSV-formatted String

        column_order: A tuple or list specifying the name of fields to be formatted as csv, in order
        missing_val: The value to be written when a named field from `column_order` is not found in the input element
        discard_extras: (bool) Behavior when additional fields are found in the dictionary input element
        dialect: Delimiters, escape-characters, etc can be controlled by providing an instance of csv.Dialect

    """

    def __init__(self, column_order, missing_val='', discard_extras=True, dialect=excel):
        self._column_order = column_order
        self._missing_val = missing_val
        self._discard_extras = discard_extras
        self._dialect = dialect

    def process(self, element, *args, **kwargs):
        result = _dict_to_csv(
			element,
            column_order=self._column_order,
            missing_val=self._missing_val,
            discard_extras=self._discard_extras,
            dialect=self._dialect
		)

        return [result,]

class DictToCSV(beam.PTransform):
    """ Transforms a PCollection of Dictionaries to a PCollection of CSV-formatted Strings

        column_order: A tuple or list specifying the name of fields to be formatted as csv, in order
        missing_val: The value to be written when a named field from `column_order` is not found in an input element
        discard_extras: (bool) Behavior when additional fields are found in the dictionary input element
        dialect: Delimiters, escape-characters, etc can be controlled by providing an instance of csv.Dialect

    """

    def __init__(self, column_order, missing_val='', discard_extras=True, dialect=excel):
        self._column_order = column_order
        self._missing_val = missing_val
        self._discard_extras = discard_extras
        self._dialect = dialect

    def expand(self, pcoll):
        return pcoll | beam.ParDo(
			_DictToCSVFn(
				column_order=self._column_order,
	            missing_val=self._missing_val,
	            discard_extras=self._discard_extras,
	            dialect=self._dialect)
            )

class FilterEmptyPollution(beam.DoFn):
	def process(self, element):
		(
			no, year, month, day, hour, pollution, dewp, temp, pressure,
			wind_dir, wind_speed, snow_hours, rain_hours
		) = element.split(',')
		if pollution == 'NA':
			pollution = 0
		yield (
			no,
			year,
			month,
			day,
			hour,
			pollution,
			dewp,
			temp,
			pressure,
			wind_dir,
			wind_speed,
			snow_hours,
			rain_hours
		)

class TimestampConverter(beam.DoFn):
	def process(self, element):
		(
			no, year, month, day, hour, pollution, dewp, temp, pressure,
			wind_dir, wind_speed, snow_hours, rain_hours
		) = element
		datetime_time = datetime(
			int(year),
			int(month),
			int(day),
			int(hour)
		)
		yield [
			no,
			str(datetime_time),
			pollution,
			dewp,
			temp,
			pressure,
			wind_dir,
			wind_speed,
			snow_hours,
			rain_hours
		]

class TypeConverter(beam.DoFn):
	def process(self, element):
		(
			no, time, pollution, dewp, temp, pressure,
			wind_dir, wind_speed, snow_hours, rain_hours
		) = element
		yield {
			'no': int(no),
			'time': time,
			'pollution': float(pollution),
			'dew_point': float(dewp),
			'temperature': float(temp),
			'pressure': float(pressure),
			'wind_dir': wind_dir,
			'wind_speed': float(wind_speed),
			'snow_hours': float(snow_hours),
			'rain_hours': float(rain_hours)
		}

class Serializer(beam.DoFn):
	def process(self, element):
		yield (
			element['no'], {
				'time': element['time'],
				'pollution': element['pollution'],
				'dew_point': element['dew_point'],
				'temperature': element['temperature'],
				'pressure': element['pressure'],
				'wind_dir': element['wind_dir'],
				'wind_speed': element['wind_speed'],
				'snow_hours': element['snow_hours'],
				'rain_hours': element['rain_hours']
			}
		)

class ShiftTimeSeries(beam.DoFn):
	def process(self, element):
		key, value = element
		yield (
			key + 1, {
				'time(t-1)': value['time'],
				'pollution(t-1)': value['pollution'],
				'dew_point(t-1)': value['dew_point'],
				'temperature(t-1)': value['temperature'],
				'pressure(t-1)': value['pressure'],
				'wind_dir(t-1)': value['wind_dir'],
				'wind_speed(t-1)': value['wind_speed'],
				'snow_hours(t-1)': value['snow_hours'],
				'rain_hours(t-1)': value['rain_hours']
			}
		)

class FilterRowsWithEmptyCols(beam.DoFn):
	def process(self, element):
		if 'pollution(t-1)' in element and 'pollution' in element:
			yield element


NUMERIC_FEATURE_KEYS = [
	'pollution',
	'dew_point',
	'temperature',
	'pressure',
	'wind_speed',
	'snow_hours',
	'rain_hours'
]

def preprocessing_fn(inputs):
	outputs = inputs.copy()
	# convert strings to integer representation
	outputs['wind_dir'] = tft.compute_and_apply_vocabulary(outputs['wind_dir'])
	# normalize features
	for key in NUMERIC_FEATURE_KEYS:
		outputs[key] = tft.scale_to_0_1(outputs[key])
	return outputs

raw_metadata = dataset_metadata.DatasetMetadata(
	dataset_schema.from_feature_spec({
		'no': tf.io.FixedLenFeature([], tf.int64),
		'time': tf.io.FixedLenFeature([], tf.string),
		'pollution': tf.io.FixedLenFeature([], tf.float32),
		'dew_point': tf.io.FixedLenFeature([], tf.float32),
		'temperature': tf.io.FixedLenFeature([], tf.float32),
		'pressure': tf.io.FixedLenFeature([], tf.float32),
		'wind_dir': tf.io.FixedLenFeature([], tf.string),
		'wind_speed': tf.io.FixedLenFeature([], tf.float32),
		'snow_hours': tf.io.FixedLenFeature([], tf.float32),
		'rain_hours': tf.io.FixedLenFeature([], tf.float32)
	}))

#options = PipelineOptions()
#p = beam.Pipeline(options=options)
p = beam.Pipeline('DirectRunner')

raw_data = (
	p |
	#TODO: Figure out a solution to skip first 24 hours
	'Read csv' >> beam.io.ReadFromText(input_filename, skip_header_lines=25) |
	'Filter empty pm2.5' >> beam.ParDo(FilterEmptyPollution()) |
	'Timestamp conversion' >> beam.ParDo(TimestampConverter()) |
	'Type conversion' >> beam.ParDo(TypeConverter())
)

raw_dataset = (raw_data, raw_metadata)

with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
	transformed_dataset, transform_fn = (
	    raw_dataset
	    | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)
	)

current_vals = (
	transformed_dataset[0] |
	'Serialize dataset' >> beam.ParDo(Serializer())
)

previous_vals = tft.beam.deep_copy.deep_copy(current_vals)

previous_vals = (
	previous_vals |
	'Shift time series' >> beam.ParDo(ShiftTimeSeries())
)

combine_results = ({'previous_vals': previous_vals, 'current_vals': current_vals} |
	beam.CoGroupByKey()
)

def join_temporal_values(element):
	joined_vals = {}
	no, values = element
	previous = values['previous_vals']
	current = values['current_vals']
	joined_vals.update({'no': no})
	[joined_vals.update(x) if x else None for x in previous]
	[joined_vals.update(x) if x else None for x in current]
	return joined_vals


csv_headers = (
	'no', 'pollution(t-1)', 'dew_point(t-1)', 'temperature(t-1)',
	'pressure(t-1)', 'wind_dir(t-1)', 'wind_speed(t-1)', 'snow_hours(t-1)',
	'rain_hours(t-1)', 'pollution'
)
csv_headerline = ','.join(csv_headers)

training_data = (
	combine_results |
	'Join values' >> beam.Map(join_temporal_values) |
	'Filter rows with empty columns' >> beam.ParDo(FilterRowsWithEmptyCols()) |
	'Format to csv' >> DictToCSV(csv_headers, missing_val='') |
	'Write csv' >> beam.io.WriteToText(
		output_filename,
		header=csv_headerline.encode()
	)
)

p.run().wait_until_finish()
