# %%
import pandas as pd
import dateutil.parser
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata

# %%

data = pd.read_csv("data_test.csv")


# %%

new_time = []
for x in data["time"]:
    new_time.append(dateutil.parser.parse(x))

data["time"] = new_time

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)
metadata.update_column(
    column_name='id',
    sdtype='id')

metadata.set_sequence_key(column_name="id")
metadata.set_sequence_index(column_name="time")
synthesizer = PARSynthesizer(metadata,verbose= True,epochs = 10000)
synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_sequences = 25)
synthetic_data.to_csv("./PAR.csv",index = False)
synthesizer.save("./PAR.pkl")


