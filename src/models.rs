/* baseline model forward:
    
    let being_model_output = being_model(being_inputs).mean(axis=0).squeeze(0);
    let fo_model_output = fo_model(fo_inputs).mean(axis=0).squeeze(0);
    let speechlet_model_output = speechlet_model(speechlet_inputs).mean(axis=0).squeeze(0);

    let intermediate = (being_model_output + fo_model_output + speechlet_output) / 3.;

    let model_output = final_output_model(intermediate);

    return model_output
*/

/* Ideally:
    set-transformer implementation for each input type, then final_output_model(intermediate) similarly.
    I remember reading something along the lines that their model subsumes sum({f(x) for all x})
*/