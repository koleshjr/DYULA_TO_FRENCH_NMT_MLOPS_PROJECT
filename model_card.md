## Description

An example of a machine translation model that translates Dyula to French using the [JoeyNMT framework](https://github.com/joeynmt/joeynmt).

## Example Payload

Here is an example payload you can use to test model inference.

```json
{
    "inputs": [
        {
            "name": "input-0",
            "shape": [1],
            "datatype": "BYTES",
            "parameters": null,
            "data": [
                "i tɔgɔ bi cogodɔ"
            ]
        }
    ]
}
```
